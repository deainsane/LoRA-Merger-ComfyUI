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
