"""Tests for building the Space's theme from config."""

import gradio as gr
import pytest

import theme


def test_palette_passes_colour_names_through():
    assert theme.palette("emerald") == "emerald"


def test_palette_from_hex_keeps_the_colour_at_500():
    colour = theme.palette("#1f6feb")
    assert isinstance(colour, gr.themes.Color)
    assert colour.c500 == "#1f6feb"
    # Lighter shades below 500, darker above.
    assert colour.c50 > colour.c400 > colour.c500
    assert colour.c600 < colour.c500
    assert colour.c950 < colour.c900 < colour.c600


@pytest.mark.parametrize("bad", ["#12345", "#gggggg", "#1f6feb00"])
def test_palette_rejects_bad_hex(bad):
    with pytest.raises(ValueError, match="hex colour"):
        theme.palette(bad)


def test_build_theme_with_defaults():
    assert isinstance(theme.build_theme(), gr.themes.Base)


def test_build_theme_with_hex_colours_and_a_font(monkeypatch):
    monkeypatch.setattr(theme, "THEME_PRIMARY_COLOUR", "#0b6e4f")
    monkeypatch.setattr(theme, "THEME_SECONDARY_COLOUR", "#f2a541")
    monkeypatch.setattr(theme, "THEME_NEUTRAL_COLOUR", "zinc")
    monkeypatch.setattr(theme, "THEME_FONT", "Inter")
    built = theme.build_theme()
    assert built.primary_500 == "#0b6e4f"
    assert "Inter" in built.font
