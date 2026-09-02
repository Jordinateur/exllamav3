# Expert-aware MoE placement (experimental)

ExLlamaV3 now supports optional per-expert CUDA placement overrides for `BlockSparseMLP` MoE layers in layer-split mode.

## What it does

Instead of placing an entire MoE layer on one GPU, you can route individual experts to explicit devices:

- `(layer_id, expert_id) -> cuda_device`
- sparse overrides are supported; unspecified experts stay on the module's default device

## Current scope

Phase-1 implementation:

- ✅ Layer-split mode
- ✅ Manual placement map
- ✅ Profile-based static map generation API
- ✅ Per-layer expert routing statistics API
- ✅ Runtime execution of remote experts with per-expert activation transfer

Not yet supported (explicitly rejected):

- ❌ Tensor-parallel mode (`tensor_p=True`)
- ❌ CPU MoE split/offload (`moe_cpu_split` / `moe_cpu_offload`)

## Manual map API

```python
config.set_expert_device_map({
    (10, 0): 0,
    (10, 1): 0,
    (10, 2): 1,
    (10, 3): 2,
})
```

Also supported:

```python
config.set_expert_device_map({
    10: {0: 0, 1: 0, 2: 1, 3: 2},
    "model.layers.11.mlp": {5: 2},
})
```

Validation:

- invalid expert ids raise an error for that layer
- invalid CUDA device ids raise an error

## Expert profiling API

```python
model.enable_expert_profiling(reset=True)
# run inference/prefill
stats = model.get_expert_statistics(normalized=True)
model.save_expert_profile("expert-profile.json", normalized=True)
```

Profile schema:

- `version`
- `model_architecture`
- `num_layers`
- `layers` (per-layer arrays of expert counts or normalized frequencies)

## Profile-based placement API

```python
profile = model.load_expert_profile("expert-profile.json")
map_ = model.plan_expert_placement_from_profile(
    profile,
    active_devices=[0, 1, 2],
    device_weights={0: 1.0, 1: 0.7, 2: 0.25},
    device_capacities={0: 22 * 1024**3, 1: 16 * 1024**3, 2: 12 * 1024**3},
    default_device=0,
)
config.set_expert_device_map(map_)
```

Manual overrides can be merged on top of profile-generated maps.

## CLI support

`model_init.add_args(...)` now includes:

- `--expert_device_map <json>`
- `--expert_profile <json>`
- `--moe_device_weights 0:1.0,1:0.7,2:0.25`
- `--moe_device_caps 0:22,1:16,2:12` (GB)

`eval/perf.py` now supports:

- `--profile_experts`
- `--expert_profile_out <json>`

## Performance notes

- Cross-device experts require activation/result transfer for routed tokens.
- This first implementation prioritizes correctness and compatibility.
- Quantized fused MoE fast paths are disabled when cross-device expert placement is active for a layer.
- Autosplit headroom can be tuned with:
  - `EXL3_MOE_EXPERT_RESERVE_MB` (default `1024`)
  - `EXL3_MOE_EXPERT_RESERVE_RATIO` (default `1.05`)
