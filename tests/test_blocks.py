from blocks import normalize_key, parse_weight_list


class TestNormalizeKey:
    def test_string_key_unchanged(self):
        assert normalize_key("diffusion_model.blocks.0.attn.wq.weight") == \
            "diffusion_model.blocks.0.attn.wq.weight"

    def test_tuple_key_uses_first_element(self):
        assert normalize_key(("diffusion_model.single_blocks.0.linear1.weight", (0, 0, 9))) == \
            "diffusion_model.single_blocks.0.linear1.weight"


class TestParseWeightList:
    def test_simple_list(self):
        assert parse_weight_list("1,1,0.8,0.5,0") == [1.0, 1.0, 0.8, 0.5, 0.0]

    def test_whitespace_tolerated(self):
        assert parse_weight_list(" 1.0 , 0.5 ") == [1.0, 0.5]

    def test_empty_token_uses_default(self):
        assert parse_weight_list("1,,0.5", default=1.0) == [1.0, 1.0, 0.5]

    def test_empty_string_returns_single_default(self):
        assert parse_weight_list("", default=1.0) == [1.0]

    def test_invalid_token_uses_default(self):
        assert parse_weight_list("1,abc,0.5", default=1.0) == [1.0, 1.0, 0.5]

    def test_none_returns_empty(self):
        assert parse_weight_list(None) == []


from blocks import make_category, key_weight

KREA2_DEF = {
    "model": "KREA2",
    "categories": [make_category("blocks", r"(?:^|\.)blocks\.(\d+)\.", 5, "1,1,1,0.8,0.5,0")],
    "pathways": [
        {"name": "txtfusion.layerwise", "regex": r"txtfusion\.layerwise_blocks\.", "weight": 0.3},
        {"name": "txtfusion.refiner", "regex": r"txtfusion\.refiner_blocks\.", "weight": 0.7},
        {"name": "txtmlp", "regex": r"(?:^|\.)txtmlp\.", "weight": 0.0},
    ],
}


class TestMakeCategory:
    def test_structure(self):
        cat = make_category("blocks", r"(?:^|\.)blocks\.(\d+)\.", 5, "1,0.5")
        assert cat == {"name": "blocks", "regex": r"(?:^|\.)blocks\.(\d+)\.",
                       "group_size": 5, "group_weights": [1.0, 0.5], "default_weight": 1.0}

    def test_group_size_clamped_to_at_least_one(self):
        assert make_category("blocks", r"x", 0, "1")["group_size"] == 1


class TestKeyWeight:
    def test_block_in_first_group(self):
        # blocks 0-4 -> group 0 -> 1.0
        assert key_weight("diffusion_model.blocks.3.attn.wq.weight", KREA2_DEF) == 1.0

    def test_block_in_fourth_group(self):
        # blocks 15-19 -> group 3 -> 0.8
        assert key_weight("diffusion_model.blocks.17.mlp.up.weight", KREA2_DEF) == 0.8

    def test_block_beyond_weight_list_uses_default(self):
        # block 40 -> group 8, list has 6 entries -> default_weight 1.0
        assert key_weight("diffusion_model.blocks.40.attn.wq.weight", KREA2_DEF) == 1.0

    def test_category_regex_does_not_match_double_blocks(self):
        # 'blocks' category must NOT catch 'double_blocks' (preceded by '_')
        assert key_weight("diffusion_model.double_blocks.2.img_attn.qkv.weight", KREA2_DEF) == 1.0

    def test_pathway_layerwise(self):
        assert key_weight("diffusion_model.txtfusion.layerwise_blocks.1.attn.wq.weight", KREA2_DEF) == 0.3

    def test_pathway_refiner(self):
        assert key_weight("diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight", KREA2_DEF) == 0.7

    def test_pathway_txtmlp(self):
        assert key_weight("diffusion_model.txtmlp.0.weight", KREA2_DEF) == 0.0

    def test_unmatched_key_is_one(self):
        assert key_weight("diffusion_model.final_layer.weight", KREA2_DEF) == 1.0


from collections import OrderedDict
from blocks import compute_lora_weights, merge_selection, apply_selection


class TestComputeLoraWeights:
    def test_only_non_default_weights_stored(self):
        keys = [
            "diffusion_model.blocks.0.attn.wq.weight",    # group 0 -> 1.0 (omitted)
            "diffusion_model.blocks.25.attn.wq.weight",   # group 5 -> 0.0 (stored)
            "diffusion_model.txtmlp.0.weight",            # pathway -> 0.0 (stored)
        ]
        out = compute_lora_weights(keys, KREA2_DEF)
        assert out == {
            "diffusion_model.blocks.25.attn.wq.weight": 0.0,
            "diffusion_model.txtmlp.0.weight": 0.0,
        }

    def test_tuple_keys_normalized(self):
        keys = [("diffusion_model.blocks.25.attn.wq.weight", (0, 0, 9))]
        out = compute_lora_weights(keys, KREA2_DEF)
        assert out == {"diffusion_model.blocks.25.attn.wq.weight": 0.0}


class TestMergeSelection:
    def test_adds_new_lora(self):
        out = merge_selection({"a": {"k": 0.5}}, "b", {"k2": 0.2})
        assert out == {"a": {"k": 0.5}, "b": {"k2": 0.2}}

    def test_override_existing_lora(self):
        out = merge_selection({"a": {"k": 0.5}}, "a", {"k2": 0.2})
        assert out == {"a": {"k2": 0.2}}

    def test_does_not_mutate_input(self):
        base = {"a": {"k": 0.5}}
        merge_selection(base, "b", {"k2": 0.2})
        assert base == {"a": {"k": 0.5}}


class TestApplySelection:
    def _stack(self):
        return OrderedDict([
            ("lora0", ["diffusion_model.blocks.25.attn.wq.weight"]),
            ("lora1", ["diffusion_model.blocks.3.attn.wq.weight"]),
        ])

    def test_selects_by_index(self):
        out = apply_selection(self._stack(), KREA2_DEF, index=0)
        assert out == {"lora0": {"diffusion_model.blocks.25.attn.wq.weight": 0.0}}

    def test_chaining_accumulates_different_indices(self):
        first = apply_selection(self._stack(), KREA2_DEF, index=0)
        second = apply_selection(self._stack(), KREA2_DEF, index=1, incoming_selection=first)
        assert set(second.keys()) == {"lora0", "lora1"}

    def test_out_of_range_index_passes_through(self):
        incoming = {"lora0": {"k": 0.5}}
        out = apply_selection(self._stack(), KREA2_DEF, index=9, incoming_selection=incoming)
        assert out == {"lora0": {"k": 0.5}}

    def test_empty_stack_passes_through(self):
        out = apply_selection(OrderedDict(), KREA2_DEF, index=0, incoming_selection={"x": {}})
        assert out == {"x": {}}


import torch
from blocks import apply_block_weights


class TestApplyBlockWeights:
    def _uda(self):
        up = torch.ones(4, 2)
        down = torch.ones(2, 3)
        return {"loraA": (up, down, torch.tensor(2.0))}

    def test_no_selection_returns_input(self):
        uda = self._uda()
        assert apply_block_weights(uda, "k", None) is uda

    def test_weight_scales_up_only(self):
        out = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.5}})
        up, down, alpha = out["loraA"]
        assert torch.allclose(up, torch.full((4, 2), 0.5))
        assert torch.allclose(down, torch.ones(2, 3))   # down untouched
        assert float(alpha) == 2.0

    def test_delta_scales_linearly(self):
        base = self._uda()["loraA"]
        base_delta = base[0] @ base[1]
        up, down, _ = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.5}})["loraA"]
        assert torch.allclose(up @ down, 0.5 * base_delta)

    def test_weight_zero_drops_lora(self):
        out = apply_block_weights(self._uda(), "k", {"loraA": {"k": 0.0}})
        assert out == {}

    def test_missing_key_defaults_to_one(self):
        out = apply_block_weights(self._uda(), "other_key", {"loraA": {"k": 0.5}})
        up, _, _ = out["loraA"]
        assert torch.allclose(up, torch.ones(4, 2))


from blocks import build_krea2_definition, build_klein_definition


class TestBuildKrea2:
    def test_shape_and_matching(self):
        d = build_krea2_definition(blocks_group_size=5, blocks_weights="1,1,1,0.8,0.5,0",
                                   txtfusion_layerwise=0.3, txtfusion_refiner=0.7, txtmlp=0.0)
        assert d["model"] == "KREA2"
        assert len(d["categories"]) == 1
        assert {p["name"] for p in d["pathways"]} == {
            "txtfusion.layerwise", "txtfusion.refiner", "txtmlp"}
        # end-to-end sanity through key_weight
        assert key_weight("diffusion_model.blocks.17.attn.wq.weight", d) == 0.8
        assert key_weight("diffusion_model.txtfusion.refiner_blocks.0.mlp.up.weight", d) == 0.7


class TestBuildKlein:
    def test_shape_and_matching(self):
        d = build_klein_definition(double_blocks_group_size=1, double_blocks_weights="1,0.5",
                                   single_blocks_group_size=5, single_blocks_weights="1,1,0")
        assert d["model"] == "FLUX.2-Klein"
        assert {c["name"] for c in d["categories"]} == {"double_blocks", "single_blocks"}
        assert d["pathways"] == []
        # double_blocks group_size 1 -> block 1 is group 1 -> 0.5
        assert key_weight("diffusion_model.double_blocks.1.img_attn.qkv.weight", d) == 0.5
        # single_blocks group_size 5 -> block 12 is group 2 -> 0.0
        assert key_weight("diffusion_model.single_blocks.12.linear1.weight", d) == 0.0
        # 'single_blocks' must not be matched by a 'double_blocks' regex and vice-versa
        assert key_weight("diffusion_model.single_blocks.0.linear1.weight", d) == 1.0
