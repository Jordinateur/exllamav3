from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LayerSelector = int | str
NormalizedExpertMap = dict[LayerSelector, dict[int, int]]


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(f"{field} must be an integer, got {value!r}")


def _normalize_layer_selector(value: Any) -> LayerSelector:
    if isinstance(value, bool):
        raise ValueError("Layer selector must be int or str, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        s = value.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        return s
    raise ValueError(f"Invalid layer selector {value!r}, expected int or non-empty str")


def normalize_expert_device_map(raw_map: Any) -> NormalizedExpertMap:
    """
    Normalize accepted map forms into:
        {layer_selector: {expert_idx: device_idx}}

    Supported inputs:
      1) {(layer_selector, expert_idx): device_idx}
      2) {layer_selector: {expert_idx: device_idx}}
    """
    if raw_map in (None, {}):
        return {}
    if not isinstance(raw_map, dict):
        raise ValueError("expert_device_map must be a dict")

    normalized: NormalizedExpertMap = {}
    for key, value in raw_map.items():
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError(f"Invalid tuple key {key!r}, expected (layer, expert)")
            layer = _normalize_layer_selector(key[0])
            expert_idx = _as_int(key[1], "Expert index")
            device_idx = _as_int(value, "Device index")
            normalized.setdefault(layer, {})[expert_idx] = device_idx
            continue

        layer = _normalize_layer_selector(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid expert_device_map value for layer {key!r}, expected dict of expert->device"
            )
        layer_map = normalized.setdefault(layer, {})
        for expert, device in value.items():
            expert_idx = _as_int(expert, "Expert index")
            device_idx = _as_int(device, "Device index")
            layer_map[expert_idx] = device_idx

    return normalized


def resolve_layer_expert_overrides(
    normalized_map: NormalizedExpertMap,
    layer_idx: int | None,
    layer_key: str | None,
) -> dict[int, int]:
    if not normalized_map:
        return {}

    merged: dict[int, int] = {}

    if layer_idx is not None and layer_idx in normalized_map:
        merged.update(normalized_map[layer_idx])
    if layer_key and layer_key in normalized_map:
        merged.update(normalized_map[layer_key])
    if layer_key and layer_key.endswith(".mlp"):
        block_key = layer_key[:-4]
        if block_key in normalized_map:
            merged.update(normalized_map[block_key])

    return merged


def validate_layer_overrides(
    overrides: dict[int, int],
    *,
    layer_name: str,
    num_experts: int,
    num_devices: int,
):
    for expert_idx, device_idx in overrides.items():
        if expert_idx < 0 or expert_idx >= num_experts:
            raise ValueError(
                f"{layer_name}: invalid expert index {expert_idx}, expected 0..{num_experts - 1}"
            )
        if device_idx < 0 or device_idx >= num_devices:
            raise ValueError(
                f"{layer_name}: invalid CUDA device {device_idx}, expected 0..{num_devices - 1}"
            )


def resolve_profile_layer(
    profile: dict[str, Any],
    *,
    layer_idx: int | None,
    layer_key: str | None,
    num_experts: int,
) -> list[float] | None:
    layers = profile.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Profile must include a dict field 'layers'")

    candidates: list[str] = []
    if layer_idx is not None:
        candidates.append(str(layer_idx))
    if layer_key:
        candidates.append(layer_key)
        if layer_key.endswith(".mlp"):
            candidates.append(layer_key[:-4])

    values = None
    for c in candidates:
        if c in layers:
            values = layers[c]
            break
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"Profile layer '{candidates[0]}' must be a list")
    if len(values) != num_experts:
        raise ValueError(
            f"Profile layer '{candidates[0]}' has {len(values)} experts, expected {num_experts}"
        )
    return [float(v) for v in values]


@dataclass
class MoeLayerPlanInput:
    layer_key: str
    layer_idx: int | None
    num_experts: int
    expert_size_bytes: int
    # Most checkpoints have identically-sized experts.  Keeping the optional per-expert sizes
    # here makes the planner correct for sliced/padded or otherwise irregular expert tensors while
    # retaining the four-argument API used by existing callers.
    expert_storage_sizes: tuple[int, ...] | None = None

    def expert_size(self, expert_idx: int) -> int:
        if self.expert_storage_sizes is not None and expert_idx < len(self.expert_storage_sizes):
            return int(self.expert_storage_sizes[expert_idx])
        return int(self.expert_size_bytes)


@dataclass
class LayerDevicePlanInput:
    """One top-level layer/module considered by the layer-split planner.

    ``storage_by_device`` is the persistent storage needed when this module's non-expert/base
    tensors are placed on a candidate device.  Expert tensors with an explicit target device are
    accounted for separately in ``fixed_storage_by_device`` passed to :func:`plan_layer_devices`.
    """

    module_key: str
    storage_by_device: dict[int, int]
    affinity_by_device: dict[int, float] = field(default_factory=dict)


def plan_layer_devices(
    layer_specs: list[LayerDevicePlanInput],
    *,
    active_devices: list[int],
    device_budgets: dict[int, int],
    fixed_storage_by_device: dict[int, int] | None = None,
    device_weights: dict[int, float] | None = None,
    headroom_bytes: int = 0,
    beam_width: int = 128,
) -> tuple[list[int], dict[int, int]]:
    """Plan base devices for a layer-split load after expert targets are known.

    This is the second pass of expert-aware loading.  The first pass fixes all explicitly targeted
    expert tensors and records their storage on the target GPUs.  This pass assigns each top-level
    module's remaining/base tensors while respecting those fixed allocations and a per-device
    budget.  A small beam search is used instead of the old online OOM/retry loop: it can keep
    enough alternatives when an early greedy choice would strand a later large module.

    The returned usage includes both fixed expert storage and the selected base-module storage.
    ``headroom_bytes`` is subtracted from every device before planning; it is not an allocator
    reservation and therefore does not double-count expert tensors at load time.
    """

    if not active_devices:
        raise ValueError("active_devices must be non-empty")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if headroom_bytes < 0:
        raise ValueError("headroom_bytes must be non-negative")

    devices = list(active_devices)
    weights = {d: 1.0 for d in devices}
    if device_weights:
        for d, w in device_weights.items():
            if d in weights:
                weights[d] = float(w)

    fixed = {d: int((fixed_storage_by_device or {}).get(d, 0)) for d in devices}
    usable = {}
    for d in devices:
        if d not in device_budgets:
            raise ValueError(f"Missing memory budget for active cuda:{d}")
        budget = int(device_budgets[d]) - int(headroom_bytes)
        if budget < 0:
            raise ValueError(f"Headroom exceeds memory budget on cuda:{d}")
        if fixed[d] > budget:
            raise RuntimeError(
                f"Expert placement requires {fixed[d] / 1024**3:.2f} GiB on cuda:{d}, "
                f"but only {budget / 1024**3:.2f} GiB is available after headroom"
            )
        usable[d] = budget - fixed[d]

    device_pos = {d: i for i, d in enumerate(devices)}
    # A lower bound for the remaining total storage lets the beam discard states that cannot
    # possibly fit all subsequent modules, without assuming a particular GPU assignment.
    suffix_min = [0] * (len(layer_specs) + 1)
    for i in range(len(layer_specs) - 1, -1, -1):
        spec = layer_specs[i]
        if not spec.storage_by_device:
            raise ValueError(f"{spec.module_key}: no candidate devices")
        suffix_min[i] = suffix_min[i + 1] + min(
            max(0, int(spec.storage_by_device.get(d, 0))) for d in devices
        )

    # State: (score, remaining tuple, previous device, path tuple).  Remaining is indexed by
    # active_devices and is sufficient to identify equivalent future states.
    # Start from the first active device, matching legacy layer-split autosplit behavior.  The
    # planner can still choose a different GPU immediately when the first module or its affinity
    # makes that preferable.
    states = [(0.0, tuple(usable[d] for d in devices), devices[0], ())]
    for i, spec in enumerate(layer_specs):
        expanded = []
        for score0, rem0, previous, path in states:
            candidates = []
            for d in devices:
                need = int(spec.storage_by_device.get(d, 0))
                if need < 0:
                    raise ValueError(f"{spec.module_key}: negative storage estimate for cuda:{d}")
                pos = device_pos[d]
                if need > rem0[pos]:
                    continue
                rem = list(rem0)
                rem[pos] -= need
                if sum(rem) < suffix_min[i + 1]:
                    continue

                affinity = float(spec.affinity_by_device.get(d, 0.0))
                # Affinity keeps an MoE layer close to the GPUs holding its hottest experts;
                # locality remains a strong tie-breaker for ordinary dense layers.
                locality = 1.0 if previous == d else 0.0
                pressure = need / max(1, usable[d])
                score = score0 + affinity * 1000.0 + locality * 2.0 \
                    + weights[d] * 0.1 - pressure * 0.01
                candidates.append((score, tuple(rem), d, path + (d,)))
            expanded.extend(candidates)

        if not expanded:
            raise RuntimeError(
                f"No feasible two-pass layer placement for module {spec.module_key!r}; "
                "reduce cache/chunk size, increase GPU budgets, or reduce expert placement pressure"
            )

        # Keep the best path for an equivalent resource state, then retain the highest-scoring
        # states. This bounds planning cost while preserving alternatives across GPUs.
        best_by_state = {}
        for state in expanded:
            key = (state[1], state[2])
            old = best_by_state.get(key)
            if old is None or state[0] > old[0]:
                best_by_state[key] = state
        states = sorted(best_by_state.values(), key = lambda s: s[0], reverse = True)[:beam_width]

    best = max(states, key = lambda s: s[0])
    path = list(best[3])
    remaining = best[1]
    usage = {
        d: fixed[d] + usable[d] - remaining[pos]
        for pos, d in enumerate(devices)
    }
    return path, usage


def allocate_experts_from_profile(
    layer_specs: list[MoeLayerPlanInput],
    profile: dict[str, Any],
    *,
    active_devices: list[int],
    device_weights: dict[int, float] | None = None,
    device_capacities: dict[int, int] | None = None,
    default_device: int | None = None,
) -> NormalizedExpertMap:
    if not active_devices:
        raise ValueError("active_devices must be non-empty")
    if default_device is None:
        default_device = active_devices[0]
    if default_device not in active_devices:
        raise ValueError(f"default_device {default_device} is not active")

    weights = {d: 1.0 for d in active_devices}
    if device_weights:
        for d, w in device_weights.items():
            if d in weights:
                weights[d] = float(w)

    usage = {d: 0 for d in active_devices}
    out: NormalizedExpertMap = {}

    # Place hottest experts first globally, then by layer.
    pending: list[tuple[float, MoeLayerPlanInput, int]] = []
    for layer in layer_specs:
        freqs = resolve_profile_layer(
            profile,
            layer_idx = layer.layer_idx,
            layer_key = layer.layer_key,
            num_experts = layer.num_experts,
        )
        if freqs is None:
            continue
        total = sum(max(0.0, f) for f in freqs)
        if total > 0:
            freqs = [max(0.0, f) / total for f in freqs]
        for expert_idx, freq in enumerate(freqs):
            pending.append((freq, layer, expert_idx))

    pending.sort(key = lambda x: x[0], reverse = True)

    for freq, layer, expert_idx in pending:
        chosen = None
        best_score = None
        for d in active_devices:
            cap = device_capacities.get(d) if device_capacities else None
            expert_size = layer.expert_size(expert_idx)
            if cap is not None and usage[d] + expert_size > cap:
                continue
            load_penalty = 0.0
            if cap:
                load_penalty = (usage[d] + expert_size) / max(1, cap)
            score = (weights[d] * (1.0 + freq)) - load_penalty
            if best_score is None or score > best_score:
                best_score = score
                chosen = d

        if chosen is None:
            cap_text = ", ".join(
                f"cuda:{d}={device_capacities[d] / 1024**3:.2f} GiB"
                for d in active_devices
                if device_capacities and d in device_capacities
            ) or "no capacities provided"
            raise RuntimeError(
                f"Cannot place expert {expert_idx} of layer {layer.layer_key!r}: "
                f"all configured expert capacities are exhausted ({cap_text})"
            )

        out.setdefault(layer.layer_idx if layer.layer_idx is not None else layer.layer_key, {})[expert_idx] = chosen
        usage[chosen] += layer.expert_size(expert_idx)

    return out


def estimate_override_bytes_per_device(
    layer_specs: list[MoeLayerPlanInput],
    normalized_map: NormalizedExpertMap,
    *,
    active_devices: list[int] | None = None,
    default_device: int | None = None,
) -> dict[int, int]:
    """
    Conservative accounting of bytes that may need to be allocated on each device due to explicit
    expert overrides. This is used to reserve room before autosplit so later module placement
    doesn't consume memory needed by relocated experts.
    """
    if active_devices is None:
        devices = sorted({
            d for layer in normalized_map.values() for d in layer.values()
        })
    else:
        devices = list(active_devices)
    out = {d: 0 for d in devices}
    if not normalized_map:
        return out

    for layer in layer_specs:
        overrides = resolve_layer_expert_overrides(
            normalized_map,
            layer_idx = layer.layer_idx,
            layer_key = layer.layer_key,
        )
        if not overrides:
            continue
        for expert_idx, device_idx in overrides.items():
            if default_device is not None and device_idx == default_device:
                continue
            if device_idx in out:
                out[device_idx] += layer.expert_size(expert_idx)
    return out
