import unittest
import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = Path(__file__).resolve().parents[1] / "exllamav3" / "model" / "moe_placement.py"
_SPEC = importlib.util.spec_from_file_location("moe_placement", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

MoeLayerPlanInput = _MOD.MoeLayerPlanInput
allocate_experts_from_profile = _MOD.allocate_experts_from_profile
normalize_expert_device_map = _MOD.normalize_expert_device_map
resolve_layer_expert_overrides = _MOD.resolve_layer_expert_overrides
resolve_profile_layer = _MOD.resolve_profile_layer
validate_layer_overrides = _MOD.validate_layer_overrides
estimate_override_bytes_per_device = _MOD.estimate_override_bytes_per_device


class TestMoeExpertPlacement(unittest.TestCase):

    def test_normalize_expert_device_map_tuple_and_nested(self):
        m = normalize_expert_device_map({
            (0, 1): 2,
            "model.layers.3.mlp": {"0": "1"},
        })
        self.assertEqual(m[0][1], 2)
        self.assertEqual(m["model.layers.3.mlp"][0], 1)

    def test_resolve_layer_expert_overrides_merges(self):
        m = normalize_expert_device_map({
            2: {0: 1},
            "model.layers.2.mlp": {1: 2},
        })
        o = resolve_layer_expert_overrides(m, 2, "model.layers.2.mlp")
        self.assertEqual(o, {0: 1, 1: 2})

    def test_validate_layer_overrides_raises(self):
        with self.assertRaises(ValueError):
            validate_layer_overrides({8: 0}, layer_name = "L", num_experts = 4, num_devices = 2)
        with self.assertRaises(ValueError):
            validate_layer_overrides({0: 3}, layer_name = "L", num_experts = 4, num_devices = 2)

    def test_resolve_profile_layer_by_index_and_key(self):
        profile = {
            "layers": {
                "3": [0.1, 0.9],
                "model.layers.4.mlp": [0.3, 0.7],
            }
        }
        self.assertEqual(
            resolve_profile_layer(profile, layer_idx = 3, layer_key = None, num_experts = 2),
            [0.1, 0.9],
        )
        self.assertEqual(
            resolve_profile_layer(
                profile,
                layer_idx = None,
                layer_key = "model.layers.4.mlp",
                num_experts = 2,
            ),
            [0.3, 0.7],
        )

    def test_allocate_experts_from_profile_prefers_weighted_device(self):
        profile = {
            "layers": {
                "0": [0.8, 0.2, 0.1, 0.05],
                "1": [0.7, 0.3, 0.2, 0.1],
            }
        }
        layers = [
            MoeLayerPlanInput("model.layers.0.mlp", 0, 4, 10),
            MoeLayerPlanInput("model.layers.1.mlp", 1, 4, 10),
        ]
        m = allocate_experts_from_profile(
            layers,
            profile,
            active_devices = [0, 1],
            device_weights = {0: 1.0, 1: 0.2},
            device_capacities = {0: 40, 1: 80},
            default_device = 0,
        )
        self.assertTrue(any(v == 1 for layer in m.values() for v in layer.values()))

    def test_estimate_override_bytes_per_device(self):
        m = normalize_expert_device_map({
            0: {1: 2, 2: 2},
            "model.layers.1.mlp": {0: 1},
        })
        specs = [
            MoeLayerPlanInput("model.layers.0.mlp", 0, 4, 100),
            MoeLayerPlanInput("model.layers.1.mlp", 1, 4, 50),
        ]
        b = estimate_override_bytes_per_device(specs, m, active_devices = [0, 1, 2])
        self.assertEqual(b[2], 200)
        self.assertEqual(b[1], 50)
        self.assertEqual(b[0], 0)

if __name__ == "__main__":
    unittest.main()
