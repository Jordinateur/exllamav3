from __future__ import annotations
from functools import cached_property
from typing import Callable
import torch
import json
import os
from .config import Config
from ..util import parse_int_list
from ..util.memory import free_mem
from .model_tp import Model_TPMixin
from .model_ls import Model_LSMixin
from ..util.tensor import g_tensor_cache
from ..cache.recurrent_util import advance_recurrent_states
from .moe_placement import (
    normalize_expert_device_map,
    MoeLayerPlanInput,
    LayerDevicePlanInput,
    allocate_experts_from_profile,
    plan_layer_devices,
    resolve_layer_expert_overrides,
    resolve_profile_layer,
    validate_layer_overrides,
)
from ..modules.block_sparse_mlp import BlockSparseMLP

class Model(Model_TPMixin, Model_LSMixin):

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        self.modules = []
        self.fwd_modules = []
        self.caps = {
            "supports_tp": True
        }
        self.active_devices = []
        self.output_device = None
        self.cache_weakrefs = {}
        self.recurrent_state_cls = None
        self.draft_verifier_params = {}

        # Index of last layer that affects KV cache, used during prefill
        self.last_kv_module_idx = None
        self.last_kv_module_idx_instance = None
        self.logit_layer_idx = None
        self.first_block_idx = None

        # Calibration options
        self.calibration_all_experts = False

        # Modules dict
        self.modules_dict = None

        # Check compatibility
        self.check_compat()


    def __iter__(self):
        for module in self.modules:
            yield from module


    def find_module(self, key: str):
        if self.modules_dict is None:
            self.modules_dict = {module.key: module for module in self}
        return self.modules_dict[key]


    @cached_property
    def _get_cache_layers(self):
        return [m for m in self if m.caps.get("kv_cache")]
    def get_cache_layers(self):
        return self._get_cache_layers


    @cached_property
    def _get_recurrent_layers(self):
        return [m for m in self if m.caps.get("recurrent_cache")]
    def get_recurrent_layers(self):
        return self._get_recurrent_layers


    def get_layer_instances(self, layer_idx):
        if not self.config.layer_map:
            return [(layer_idx, 0)]
        return [
            (layer_idx, instance)
            for instance in range(self.config.layer_map.count(layer_idx))
        ]


    def num_unmapped_layers(self):
        indices = set()
        for m in self.modules:
            if m.layer_idx is not None:
                assert m.layer_idx not in indices
                indices.add(m.layer_idx)
        assert len(indices) == max(indices) + 1
        return len(indices)


    def prepare_layer_map(self):
        """
        Prepare list of (module, instance_num, original_idx) for mapped layers
        """

        # Parse layer map string
        if self.config.layer_map is None and self.config.layer_map_str:
            self.config.layer_map = parse_int_list(
                self.config.layer_map_str,
                min_value = 0,
                max_value = self.num_unmapped_layers() - 1
            )

        # No layer map
        if not self.config.layer_map:
            self.fwd_modules = [(m, 0, idx) for idx, m in enumerate(self.modules)]
            self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)
            return

        # Isolate and enumerate indexed layers
        prolog = []
        inner = []
        epilog = []
        for m in self.modules:
            if m.layer_idx is None:
                if not inner:
                    prolog.append(m)
                else:
                    epilog.append(m)
            else:
                assert len(inner) == m.layer_idx, "Inner layers are not in order"
                assert not epilog, "Inner layers are not consecutive"
                inner.append(m)

        # Compile relayered list
        relayered = []
        inner_count = [0] * len(inner)
        offset = 0
        for idx, m in enumerate(prolog):
            relayered.append((m, 0, idx + offset))
        offset = len(prolog)
        for idx in self.config.layer_map:
            relayered.append((inner[idx], inner_count[idx], idx + offset))
            inner_count[idx] += 1
        offset = len(prolog) + len(inner)
        for idx, m in enumerate(epilog):
            relayered.append((m, 0, idx + offset))

        self.fwd_modules = relayered
        if self.last_kv_module_idx is not None:
            self.last_kv_module_idx_instance = (
                self.last_kv_module_idx,
                inner_count[self.last_kv_module_idx - len(prolog)] - 1
            )


    @staticmethod
    def from_config(
        config: Config,
        component: str = "text",
        **kwargs
    ):
        """
        Create model instance from config

        :param config:
            Config created with Config.from_directory()

        :param component:
            Which component model to load, for models with multiple component.
        """

        assert component in config.model_classes, \
            f"{config.architecture} does not define a '{component}' component model"

        model = config.model_classes[component](config, **kwargs)
        model.component = component

        # Compile layer map after model is constructed (before any caches are attached)
        model.prepare_layer_map()
        return model


    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Overridden by model arch class
        raise NotImplementedError()


    @torch.inference_mode
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        """
        Run prompt-prefill inference and update cache/recurrent state.

        Inputs are first normalized by the architecture-specific prepare_inputs() hook. If the model is loaded in
        tensor-parallel mode, execution is dispatched through prefill_tp(), which fans the work out to TP workers
        and stops at the last module that affects K/V cache state. Otherwise the layer-split path prefill_ls() runs
        directly in the current process. Both paths advance recurrent states after the forward work completes.
        """
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        if self.loaded_tp:
            y = self.prefill_tp(x, params, self.last_kv_module_idx, self.modules)
            advance_recurrent_states(input_ids, params, self)
            return y
        else:
            y = self.prefill_ls(x, params)
            advance_recurrent_states(input_ids, params, self)
            return y


    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        """
        Run a normal model forward pass for generation or verification.

        After architecture-specific input preparation, tensor-parallel models dispatch through forward_tp() so each
        worker processes its shard and gathers outputs as needed. Non-TP models use the local layer-split forward
        path forward_ls(). Recurrent state advancement is shared between the two modes and runs after the selected
        forward path returns.
        """
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        if self.loaded_tp:
            y = self.forward_tp(x, params, self.last_kv_module_idx, self.modules)
            advance_recurrent_states(input_ids, params, self)
            return y
        else:
            y = self.forward_ls(x, params)
            advance_recurrent_states(input_ids, params, self)
            return y


    def unload(self):
        for module in self.modules:
            module.unload()
        self.active_devices = []
        self.unload_tp()
        self.output_device = None
        # Attached caches lose their layer tensors with the modules that allocated them
        for ref in self.cache_weakrefs.values():
            cache = ref()
            if cache is not None:
                cache.initialized = False


    def enable_expert_profiling(self, reset: bool = True):
        for module in self:
            if isinstance(module, BlockSparseMLP):
                module.enable_expert_profiling(reset = reset)


    def disable_expert_profiling(self):
        for module in self:
            if isinstance(module, BlockSparseMLP):
                module.enable_expert_profiling(False)


    def get_expert_statistics(self, normalized: bool = False) -> dict:
        layers = {}
        for module in self:
            if isinstance(module, BlockSparseMLP):
                stats = module.get_expert_statistics(normalized = normalized)
                if stats is not None:
                    layers[module.key] = stats
        return {
            "version": 1,
            "model_architecture": self.config.architecture,
            "num_layers": self.config.num_hidden_layers,
            "layers": layers,
        }


    def reset_expert_statistics(self):
        for module in self:
            if isinstance(module, BlockSparseMLP):
                module.reset_expert_statistics()


    def save_expert_profile(self, path: str, normalized: bool = True):
        with open(path, "w", encoding = "utf8") as f:
            json.dump(self.get_expert_statistics(normalized = normalized), f, indent = 2)


    def load_expert_profile(self, path: str) -> dict:
        with open(path, "r", encoding = "utf8") as f:
            profile = json.load(f)
        return profile


    def plan_expert_placement_from_profile(
        self,
        profile: dict | str,
        *,
        active_devices: list[int],
        device_weights: dict[int, float] | None = None,
        device_capacities: dict[int, int] | None = None,
        default_device: int | None = None,
    ) -> dict:
        if isinstance(profile, str):
            with open(profile, "r", encoding = "utf8") as f:
                profile = json.load(f)
        specs = self._moe_layer_specs()
        return allocate_experts_from_profile(
            specs,
            profile,
            active_devices = active_devices,
            device_weights = device_weights,
            device_capacities = device_capacities,
            default_device = default_device,
        )


    def _moe_layer_specs(self) -> list[MoeLayerPlanInput]:
        specs = []
        for module in self:
            if isinstance(module, BlockSparseMLP):
                specs.append(MoeLayerPlanInput(
                    layer_key = module.key,
                    layer_idx = module.infer_layer_idx(),
                    num_experts = module.num_experts,
                    expert_size_bytes = module.estimate_expert_storage_size(),
                    expert_storage_sizes = tuple(module.estimate_expert_storage_sizes()),
                ))
        return specs


    @staticmethod
    def _module_storage_size(module, excluded_ids: set[int] | frozenset[int] = frozenset()) -> int:
        """Estimate persistent storage for a module without double-counting composite modules.

        Most composite modules expose storage through their own ``storage_size`` implementation,
        while ``BlockSparseMLP`` deliberately has no aggregate implementation because expert
        tensors can live on different devices.  When expert leaves are excluded, recurse through
        the composite and count only the remaining leaves.  Attached KV/recurrent states are
        included here because they are allocated by the module's ``load_local`` path and are not
        part of ``Module.modules``.
        """

        excluded_ids = frozenset(excluded_ids)
        contains_cache = {}

        def has_excluded(node) -> bool:
            ident = id(node)
            if ident in contains_cache:
                return contains_cache[ident]
            result = ident in excluded_ids or any(
                has_excluded(child) for child in getattr(node, "modules", ())
            )
            contains_cache[ident] = result
            return result

        def attached_storage(node) -> int:
            total = 0
            seen = set()
            for attr in ("cache_layers", "recurrent_layers"):
                for state in getattr(node, attr, ()) or ():
                    if id(state) in seen:
                        continue
                    seen.add(id(state))
                    fn = getattr(state, "storage_size", None)
                    if callable(fn):
                        total += int(fn())
            return total

        def visit(node) -> int:
            if id(node) in excluded_ids:
                return 0
            children = tuple(getattr(node, "modules", ()) or ())
            fn = getattr(node, "storage_size", None)
            # A composite aggregate is valid only when no excluded expert is below it.  Otherwise
            # recurse so the expert leaves can be removed from the estimate.
            if callable(fn) and (not children or not has_excluded(node)):
                return int(fn()) + attached_storage(node)
            if not children:
                return (int(fn()) if callable(fn) else 0) + attached_storage(node)
            return attached_storage(node) + sum(visit(child) for child in children)

        return visit(module)


    @staticmethod
    def _device_load_budgets(
        active_devices: list[int],
        reserve_per_device: list[int] | None,
        use_per_device: list[int] | None,
    ) -> dict[int, int]:
        """Return the same cumulative allocator budgets used by the load helpers."""

        budgets = {}
        for device in active_devices:
            current = int(torch.cuda.memory_reserved(device))
            if reserve_per_device is not None:
                free, _ = torch.cuda.mem_get_info(device)
                budget = current + int(free) - int(reserve_per_device[device])
            elif use_per_device is not None:
                budget = current + int(use_per_device[device])
            else:
                raise RuntimeError("Logic error: one of reserve_per_device/use_per_device is required")
            budgets[device] = max(0, int(budget))
        return budgets


    def _plan_expert_aware_layers(
        self,
        modules: list,
        active_devices: list[int],
        reserve_per_device: list[int] | None,
        use_per_device: list[int] | None,
        verbose: bool,
    ) -> list[int]:
        """Run the two-pass expert-aware layer planner.

        Pass one is the profile/manual expert map already constructed in ``load_gen``.  This pass
        converts that map into fixed per-device expert storage and candidate base-module costs.
        ``plan_layer_devices`` then selects the base device for every top-level module before any
        CUDA weight is materialized.
        """

        normalized_map = normalize_expert_device_map(
            getattr(self.config.infer_params, "expert_device_map", {})
        )
        if not normalized_map:
            return [active_devices[0]] * len(modules)

        num_devices = torch.cuda.device_count()
        fixed_storage = {d: 0 for d in active_devices}
        layer_specs = []

        profile = getattr(self.config.infer_params, "expert_profile", None)
        if isinstance(profile, str):
            with open(profile, "r", encoding = "utf8") as f:
                profile = json.load(f)

        for module in modules:
            moe_modules = [m for m in module if isinstance(m, BlockSparseMLP)]
            expert_modules = set()
            for moe in moe_modules:
                expert_modules.update(id(x) for x in (
                    (moe.gates if moe.gated else []) + moe.ups + moe.downs
                ))

            base_storage = self._module_storage_size(module, expert_modules)
            storage_by_device = {}
            affinity_by_device = {}

            # CPU-preferred modules do not consume CUDA persistent storage.  They still appear in
            # the top-level sequence so the second-pass path can preserve their ordering.
            if module.caps.get("prefer_cpu"):
                if moe_modules:
                    raise NotImplementedError(
                        f"{module.key}: expert-aware placement cannot target a CPU-preferred MoE module"
                    )
                for device in active_devices:
                    storage_by_device[device] = 0
                    affinity_by_device[device] = 0.0
                layer_specs.append(LayerDevicePlanInput(
                    module_key = module.key,
                    storage_by_device = storage_by_device,
                    affinity_by_device = affinity_by_device,
                ))
                continue

            # Fixed expert tensors are counted once, independently of the module's eventual base
            # device. Experts without an explicit target remain part of the candidate base cost.
            local_affinity = {d: 0.0 for d in active_devices}
            total_affinity = 0.0
            for moe in moe_modules:
                overrides = resolve_layer_expert_overrides(
                    normalized_map, moe.infer_layer_idx(), moe.key
                )
                validate_layer_overrides(
                    overrides,
                    layer_name = moe.key,
                    num_experts = moe.num_experts,
                    num_devices = num_devices,
                )
                expert_sizes = moe.estimate_expert_storage_sizes()
                if len(expert_sizes) != moe.num_experts:
                    raise ValueError(
                        f"{moe.key}: expert-aware placement requires all experts to be materialized "
                        f"({len(expert_sizes)} local, {moe.num_experts} declared)"
                    )
                freqs = None
                if isinstance(profile, dict):
                    freqs = resolve_profile_layer(
                        profile,
                        layer_idx = moe.infer_layer_idx(),
                        layer_key = moe.key,
                        num_experts = moe.num_experts,
                    )
                if freqs is None:
                    freqs = [1.0] * len(expert_sizes)
                freqs = [max(0.0, float(x)) for x in freqs]
                freq_total = sum(freqs)
                if freq_total <= 0.0:
                    freqs = [1.0] * len(expert_sizes)
                    freq_total = float(len(expert_sizes))
                for expert_idx, expert_size in enumerate(expert_sizes):
                    affinity = freqs[expert_idx] / freq_total
                    total_affinity += affinity
                    target = overrides.get(expert_idx)
                    if target is None:
                        continue
                    if target not in fixed_storage:
                        raise ValueError(
                            f"{moe.key}: expert {expert_idx} targets inactive cuda:{target}"
                        )
                    fixed_storage[target] += expert_size
                    local_affinity[target] += affinity

            for device in active_devices:
                candidate_storage = base_storage
                candidate_affinity = local_affinity[device]
                for moe in moe_modules:
                    overrides = resolve_layer_expert_overrides(
                        normalized_map, moe.infer_layer_idx(), moe.key
                    )
                    expert_sizes = moe.estimate_expert_storage_sizes()
                    freqs = None
                    if isinstance(profile, dict):
                        freqs = resolve_profile_layer(
                            profile,
                            layer_idx = moe.infer_layer_idx(),
                            layer_key = moe.key,
                            num_experts = moe.num_experts,
                        )
                    if freqs is None:
                        freqs = [1.0] * len(expert_sizes)
                    freqs = [max(0.0, float(x)) for x in freqs]
                    freq_total = sum(freqs)
                    if freq_total <= 0.0:
                        freqs = [1.0] * len(expert_sizes)
                        freq_total = float(len(expert_sizes))
                    for expert_idx, expert_size in enumerate(expert_sizes):
                        if expert_idx not in overrides:
                            candidate_storage += expert_size
                            candidate_affinity += freqs[expert_idx] / freq_total
                storage_by_device[device] = int(candidate_storage)
                affinity_by_device[device] = (
                    candidate_affinity / max(1.0, total_affinity)
                )

            layer_specs.append(LayerDevicePlanInput(
                module_key = module.key,
                storage_by_device = storage_by_device,
                affinity_by_device = affinity_by_device,
            ))

        budgets = self._device_load_budgets(
            active_devices, reserve_per_device, use_per_device
        )
        headroom_mb = int(os.environ.get("EXL3_MOE_PLAN_HEADROOM_MB", "512"))
        device_plan, usage = plan_layer_devices(
            layer_specs,
            active_devices = active_devices,
            device_budgets = budgets,
            fixed_storage_by_device = fixed_storage,
            device_weights = getattr(self.config.infer_params, "moe_device_weights", None),
            headroom_bytes = max(0, headroom_mb) * 1024**2,
        )

        if verbose:
            fixed_gb = ", ".join(
                f"cuda:{d}={fixed_storage[d] / 1024**3:.2f} GiB"
                for d in active_devices
            )
            usage_gb = ", ".join(
                f"cuda:{d}={usage[d] / 1024**3:.2f} GiB"
                for d in active_devices
            )
            print(f" -- Expert pass: fixed expert storage ({fixed_gb})")
            print(f" -- Layer pass: planned storage ({usage_gb})")
            print(" -- Layer pass: " + ", ".join(
                f"{modules[i].key}->cuda:{device_plan[i]}"
                for i in range(len(modules))
            ))

        return device_plan


    def load_gen(
        self,
        device: torch.device | str | int | None = None,
        tp_output_device: torch.device | str | int | None = None,
        reserve_per_device: list[float] | float | None = None,
        use_per_device: list[float] | float | None = None,
        tensor_p: bool = False,
        progressbar: bool = False,
        max_chunk_size: int = 2048,
        max_output_size: int = 32,
        max_output_factor: int = 1,
        callback: Callable[[int, int], None] | None = None,
        generator: bool = True,
        tp_dev_limits: dict | None = None,
        tp_backend: str = "native",
        verbose: bool = False,
        max_batch_size: int = 1,
        tp_options: dict | None = None,
        autosplit_no_forward: bool = False,
    ):
        """
        Load model, generator function. For regular function, call load() with the same arguments

        :param device:
            (optional) If specified, load to single device, e.g. "cuda:0"

        :param tp_output_device:
            (optional) If loading with tensor_p == True, device on which to gather output logits. Must be one of
            the active devices in the split. Default is first device in split

        :param reserve_per_device:
            (optional) Amount of memory to reserve for any device. Either a value in GB to apply on all devices
            or a list of floats giving an individual reserve per device. Negative reserve excludes device from
            split. E.g.:

            # reserve 4.5 GB on cuda:0, 1 GB on each cuda:1 and on cuda:2
            model.load(reserve_per_device = [4.5, 1, 1])

            # reserve 1 GB on cuda:0 and cuda:2, exclude cuda:1
            model.load(reserve_per_device = [1, -1, 1])

            The default reserve per device is 0.5 GB. This applies to devices not included in reserve_per_device
            as well.

        :param use_per_device:
            (optional) Amount of memory to use per device.

            Does not account for memory allocated by other processes or by the calling process up to the call
            to model.load(), i.e. if cuda:0 currently has 3 GB in use and user_per_device = [12, ...], at the
            end of loading cuda:0 will have up to 15 GB of VRAM allocated, using up to 15 GB during a forward
            pass.

            Devices not included in use_per_device, or included with a value of 0, will not be used, e.g.:

            # use up to 23 GB on cuda:0 and cuda:2, do not load on cuda:1 and cuda:3 (if present)
            model.load(use_per_device = [23, 0, 23])

        :param tensor_p:
            Load in tensor-parallel mode. By default, attempt to split model according to available VRAM.
            Allocation can be overridden with use_per_device or modified by reserve_per_device.

        :param max_chunk_size:
            The maximum number of tokens to expect in a single forward pass. Informs the layer split only, and
            makes no difference when loading on a single device.

        :param max_output_size:
            The maximum number of output tokens to expect in a single forward pass. Informs the estimate of the
            size of the output logits. Values larger than max_chunk_size have no effect.

        :param max_output_factor:
            When estimating the memory footprint of the output layer, scale the size of the output tensor by
            this factor. For instance, if the first thing you wish to do with a float16 output tensor is upcast
            to float32, a value of 3 here would (attempt to) make sure the output layer always ends up on a
            device where there is enough space for that.

        :param progressbar:
            Show rich progressbar while loading

        :param callback:
            If provided, called with (current_module, num_modules) for every module loaded. Don't specify a
            callback function when using the

        :param generator:
            Always true when using the _gen function directly

        :param tp_dev_limits:
            (optional, TP only) Dictionary of module categories and max parallelism for each. Categories are
            "mlp", "attn", "moe", "linear" (i.e. output layer). Example:
            tp_dev_limits = {
                "attn": 2,  # Each attn layer uses at most two devices for tensor parallelism
                "moe": 3,  # etc.
            }

        :param tp_backend:
            str, either "nccl" (default) or "native"

        :param verbose:
            bool, more info while loading including full TP split

        :param max_batch_size:
            Max batch size to account for when loading in autosplit mode (default: 1)

        :param tp_options:
            dict of optional values:
                "moe_tensor_split": bool - use tensor split rather than expert parallelism for MoE layers

        :param autosplit_no_forward:
            For debug purposes, skip reference forward pass during autosplit load.
        """

        free_mem()

        # Route CPU-offloaded MoE layers to this component's own worker and budget (an MTP head
        # shares the config but loads after the main model's worker has already started)
        self.config.infer_params.moe_cpu_component = getattr(self, "component", "text")
        self.config.infer_params.expert_device_map = normalize_expert_device_map(
            getattr(self.config.infer_params, "expert_device_map", {})
        )
        self.config.expert_device_map = self.config.infer_params.expert_device_map

        assert not (bool(reserve_per_device) and bool(use_per_device)), \
            "Cannot specify both memory usage and memory reserve."

        if tensor_p and (self.config.infer_params.expert_device_map or self.config.infer_params.expert_profile):
            raise NotImplementedError("expert-aware MoE placement is currently supported in layer-split mode only")
        if self.config.infer_params.expert_device_map and (
            self.config.infer_params.moe_cpu_offload or
            self.config.infer_params.moe_cpu_split or
            self.config.infer_params.draft_moe_cpu_offload
        ):
            raise NotImplementedError("expert_device_map is not supported with moe_cpu_offload/moe_cpu_split")

        assert max_chunk_size >= 1, "max_chunk_size must be positive"
        assert max_output_size >= 1, "max_output_size must be positive"
        assert max_output_factor >= 1, "max_output_factor must be positive"

        # Load to single device
        if device is not None:
            assert not bool(reserve_per_device) and not bool(use_per_device), \
                "Cannot specify reserve_per_device or use_per_device when loading to single device."
            assert not tensor_p, \
                "Cannot use tensor_p when loading to single device."
            if self.config.infer_params.expert_profile:
                dev_idx = torch.device(device).index or 0
                prof_map = self.plan_expert_placement_from_profile(
                    self.config.infer_params.expert_profile,
                    active_devices = [dev_idx],
                    device_weights = self.config.infer_params.moe_device_weights,
                    device_capacities = self.config.infer_params.moe_device_capacities,
                    default_device = dev_idx,
                )
                merged = normalize_expert_device_map(self.config.infer_params.expert_device_map)
                for lk, lv in prof_map.items():
                    merged.setdefault(lk, {}).update(lv)
                self.config.infer_params.expert_device_map = merged
                self.config.expert_device_map = merged
            self._load_single(progressbar, device, self.config, self.modules, verbose)
            self.output_device = self.modules[-1].device

        # Use/reserve
        else:
            rpd = reserve_per_device is not None
            upd = use_per_device is not None
            assert not (rpd and upd), \
                "Cannot specify both reserve_per_device or use_per_device."
            num_devices = torch.cuda.device_count()
            layer_device_plan = None

            if not upd:
                if reserve_per_device is None:
                    reserve_per_device = [0.5] * num_devices
                elif any(isinstance(reserve_per_device, t) for t in [float, int]):
                    reserve_per_device = [reserve_per_device] * num_devices
                elif not isinstance(reserve_per_device, list):
                    raise ValueError("reserve_per_device must be float or list[float]")
                while len(reserve_per_device) < num_devices:
                    reserve_per_device.append(0.5)
                reserve_per_device = [int(x * 1024**3) for x in reserve_per_device]
                active_devices = [
                    i for i in range(num_devices)
                    if i >= len(reserve_per_device) or reserve_per_device[i] >= 0
                ]

            if upd:
                if any(isinstance(use_per_device, t) for t in [float, int]):
                    use_per_device = [use_per_device] * num_devices
                elif not isinstance(use_per_device, list):
                    raise ValueError("use_per_device must be float or list[float]")
                use_per_device = [int(x * 1024**3) for x in use_per_device]
                active_devices = [
                    i for i, x in enumerate(use_per_device)
                    if x > 0
                ]

            if self.config.infer_params.expert_profile:
                profile_default_device = active_devices[0] if active_devices else None
                prof_map = self.plan_expert_placement_from_profile(
                    self.config.infer_params.expert_profile,
                    active_devices = active_devices,
                    device_weights = self.config.infer_params.moe_device_weights,
                    device_capacities = self.config.infer_params.moe_device_capacities,
                    default_device = profile_default_device,
                )
                merged = normalize_expert_device_map(self.config.infer_params.expert_device_map)
                for lk, lv in prof_map.items():
                    merged.setdefault(lk, {}).update(lv)
                self.config.infer_params.expert_device_map = merged
                self.config.expert_device_map = merged

            # Expert-aware loading is planned before any CUDA module is materialized.  The planner
            # accounts fixed expert targets and chooses a base device for every top-level module;
            # the loader then executes that plan in its second pass.
            if self.config.infer_params.expert_device_map and not tensor_p:
                layer_device_plan = self._plan_expert_aware_layers(
                    self.modules,
                    active_devices,
                    reserve_per_device,
                    use_per_device,
                    verbose,
                )

            # Split load
            if not tensor_p:
                yield from self._load_autosplit(
                    progressbar,
                    reserve_per_device,
                    use_per_device,
                    active_devices,
                    max_chunk_size,
                    max_output_size,
                    max_output_factor,
                    callback,
                    generator,
                    self.config,
                    self.modules,
                    verbose,
                    max_batch_size,
                    self.cache_weakrefs,
                    autosplit_no_forward,
                    layer_device_plan,
                )
                self.output_device = self.modules[-1].device

            # Tensor-P load:
            else:
                if not self.caps.get("supports_tp"):
                    raise NotImplementedError(f"Tensor-parallel is not currently implemented for {self.config.architecture}")
                if self.config.layer_map:
                    raise NotImplementedError(f"Tensor-parallel is not currently implemented for relayered models.")

                if tp_output_device is None:
                    tp_output_device = active_devices[0]
                else:
                    assert torch.device(tp_output_device).index in active_devices, \
                        "Output device must be part of split."

                if tp_options is None:
                    tp_options = {}

                yield from self._load_tp(
                    progressbar,
                    reserve_per_device,
                    use_per_device,
                    active_devices,
                    max_chunk_size,
                    max_output_size,
                    max_output_factor,
                    callback,
                    generator,
                    tp_output_device,
                    self.config,
                    self.modules,
                    tp_dev_limits,
                    tp_backend,
                    verbose,
                    tp_options,
                )
                self.output_device = tp_output_device

        free_mem()

        # Release all global shared tensors (refs still held by modules until model is unloaded)
        g_tensor_cache.drop_all()

        # Mark every attached cache usable.
        for ref in self.cache_weakrefs.values():
            cache = ref()
            if cache is not None:
                cache.initialized = True


    @torch.inference_mode
    def load(self, *args, **kwargs):
        """
        Load as a regular function, see arguments for load_gen().
        """

        kwargs["generator"] = False
        f = self.load_gen(*args, **kwargs)
        for _ in f: pass

        # CPU-offloaded MoE experts: make sure the worker processes are up before the first
        # forward (spawned during module loading when the offload count is exact, so the child's
        # weight loading overlaps the parent's; started here otherwise). ensure_started is
        # idempotent, so already-running workers of other components are unaffected
        for host in getattr(self.config, "moe_cpu_hosts", {}).values():
            host.ensure_started()


    def get_load_metrics(self):
        return self.config.stc.get_metrics()


    def get_layout_tree(self, pre_indent: int) -> str:
        def get_branch(module, b_indent) -> str:
            nonlocal pre_indent
            lines = [get_branch(m, b_indent + 4) for m in module.modules]
            dedup_lines = []
            count = 1
            for i in range(len(lines)):
                if i < len(lines) - 1 and lines[i] == lines[i + 1]:
                    count += 1
                else:
                    pref = ""
                    if count > 1:
                        pref = f"[{count}x] "
                        count = 1
                    dedup_lines.append(lines[i].replace("[]", pref))
            r = " " * (pre_indent + b_indent) + " - []" + module.get_name() + "\n"
            r += "".join(dedup_lines)
            return r

        def compact_rle(s: list[str]) -> list[tuple[int, list[str]]]:
            n = len(s)
            if n == 0:
                return []
            dp = [(0, None)] + [(float('inf'), None)] * n
            for i in range(1, n + 1):
                for j in range(i - 1, -1, -1):
                    seg = s[j:i]
                    seg_len = len(seg)
                    p = _smallest_period(seg)
                    if p is not None:
                        k = seg_len // p
                        cost = dp[j][0] + p
                        if cost < dp[i][0]:
                            dp[i] = (cost, (j, k, s[j:j + p]))
            result = []
            pos = n
            while pos > 0:
                _, (j, k, pattern) = dp[pos]
                result.append((k, pattern))
                pos = j
            result.reverse()
            return result

        def _smallest_period(seg: list[str]) -> int | None:
            n = len(seg)
            for p in range(1, n + 1):
                if n % p == 0 and all(seg[i] == seg[i % p] for i in range(p, n)):
                    return p
            return None

        br = get_branch(self, 0).replace("[]", "").rstrip()
        br = br.split("\n")
        cbr = compact_rle(br)
        br = []
        for num, span in cbr:
            if num == 1:
                br += span
            else:
                br += [" " * span[0].index("- ") + f"- [{num}x]"]
                br += ["    " + s for s in span]

        return "\n".join(br)


    def get_storage_info(self):
        from ..modules import Linear
        def get_tensor_size(tensors):
            return 8 * sum(t.element_size() * t.numel() for t in tensors.values())
        sum_bits = 0
        sum_numel = 0
        head_bpw = 0
        head_numel = 0
        for module in self:
            if module.key.endswith("lm_head"):
                if module.device is not None:
                    head_bpw = get_tensor_size(module.get_tensors()) / module.weights_numel()
                else:
                    head_bpw = sum(self.config.stc.get_tensor_sizes(module.key)) / module.weights_numel() * 8
                head_numel = module.weights_numel()
            elif isinstance(module, Linear):
                if module.device is not None:
                    sum_bits += get_tensor_size(module.get_tensors())
                else:
                    sum_bits += sum(self.config.stc.get_tensor_sizes(module.key)) * 8
                sum_numel += module.weights_numel()
        vram_bits = head_numel * head_bpw + sum_bits
        return sum_bits / sum_numel, head_bpw, vram_bits


    def get_name(self):
        return self.__class__.__name__


    @staticmethod
    def get_additional_compiled_tensors(config: Config):
        return {}


    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """
        Convenience function for formatting a single chat request with the default template associated with the
        model's architecture, to simplify example and test scripts. Doesn't consider the model's actual Jinja template.
        """
        raise NotImplementedError()


    def batch_recurrent_states(self):
        raise NotImplementedError()


    def check_compat(self):
        """
        Decide if any model-specific requirements are met when creating Model
        """
        pass
