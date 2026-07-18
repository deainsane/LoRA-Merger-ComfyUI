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
