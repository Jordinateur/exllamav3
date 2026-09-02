from __future__ import annotations

from dataclasses import dataclass
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
            if cap is not None and usage[d] + layer.expert_size_bytes > cap:
                continue
            load_penalty = 0.0
            if cap:
                load_penalty = (usage[d] + layer.expert_size_bytes) / max(1, cap)
            score = (weights[d] * (1.0 + freq)) - load_penalty
            if best_score is None or score > best_score:
                best_score = score
                chosen = d

        if chosen is None:
            chosen = default_device

        if chosen != default_device:
            out.setdefault(layer.layer_idx if layer.layer_idx is not None else layer.layer_key, {})[expert_idx] = chosen
        usage[chosen] += layer.expert_size_bytes

    return out

