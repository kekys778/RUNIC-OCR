"""Unit tests for the rune <-> Rundata transliteration tables (no GPU / ML deps)."""

import pytest

from runic_transliteration import (
    ANGLO_SAXON_FUTHORC_MAP,
    ELDER_FUTHARK_MAP,
    TranslitConfig,
    translit_to_rune_string,
    transliterate,
    transliterate_with_spaces,
)


@pytest.mark.parametrize(
    ("runic", "expected"),
    [
        ("ᚠᚢᚦᚨᚱᚲ", "fuþark"),
        ("ᚨᛚᚢ", "alu"),
        ("ᛚᚨᚢᚲᚨᛉ", "laukaR"),
        ("ᛏᛁᚹᚨᛉ", "tiwaR"),
        ("ᛁᚾᚷᚹᚨᛉ", "ingwaR"),
    ],
)
def test_elder_futhark_words(runic, expected):
    assert transliterate(runic) == expected


def test_word_dividers_are_kept():
    assert transliterate("ᚠᚢᚦᚨᚱᚲ᛬ᚨᛚᚢ") == "fuþark:alu"


def test_spaced_output():
    assert transliterate_with_spaces("ᚨᛚᚢ") == "a l u"


def test_unknown_symbol_marker():
    assert transliterate("ᚠXᚢ") == "f?u"


def test_younger_futhark():
    assert transliterate("ᚼᛅᛁ", TranslitConfig(alphabet="younger")) == "hai"


def test_reverse_mapping():
    assert translit_to_rune_string("alu") == "ᚨᛚᚢ"


@pytest.mark.parametrize("rune", sorted(ELDER_FUTHARK_MAP))
def test_elder_round_trip_per_rune(rune):
    """Every single Elder Futhark rune survives rune -> latin -> rune."""
    assert translit_to_rune_string(transliterate(rune)) == rune


def test_futhorc_os_rune():
    assert ANGLO_SAXON_FUTHORC_MAP["ᚩ"] == "o"


def test_config_is_immutable():
    with pytest.raises(AttributeError):
        TranslitConfig().separator = " "  # type: ignore[misc]
