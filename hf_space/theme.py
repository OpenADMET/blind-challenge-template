"""Build the Space's Gradio theme from the colours and font in ``config.py``."""

import re
from typing import Any

import gradio as gr
from config import (
    THEME_FONT,
    THEME_NEUTRAL_COLOUR,
    THEME_PRIMARY_COLOUR,
    THEME_SECONDARY_COLOUR,
)
from gradio.themes.utils import sizes

HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")

# How far each shade is mixed towards white (lighter shades) or black (darker
# shades); 500 is the colour itself. Roughly matches Gradio's built-in palettes.
_SHADE_MIX = {
    "c50": ("white", 0.92),
    "c100": ("white", 0.82),
    "c200": ("white", 0.64),
    "c300": ("white", 0.44),
    "c400": ("white", 0.22),
    "c500": ("white", 0.0),
    "c600": ("black", 0.14),
    "c700": ("black", 0.3),
    "c800": ("black", 0.45),
    "c900": ("black", 0.6),
    "c950": ("black", 0.72),
}


def _mix(hex_colour: str, towards: str, amount: float) -> str:
    """Mix a hex colour towards white or black by ``amount`` (0 to 1)."""
    target = 255 if towards == "white" else 0
    channels = [int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(c + (target - c) * amount) for c in channels]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def palette(colour: str) -> gr.themes.Color | str:
    """Turn a colour setting into something a Gradio theme accepts.

    Parameters
    ----------
    colour : str
        A Gradio colour name, such as "blue" or "emerald", or a hex colour such as
        "#1f6feb", from which lighter and darker shades are made.

    Returns
    -------
    gr.themes.Color | str
        The palette for a hex colour, or the name unchanged.

    Raises
    ------
    ValueError
        If ``colour`` starts with "#" but isn't a six-digit hex colour.

    """
    if not colour.startswith("#"):
        return colour
    if not HEX_COLOUR.match(colour):
        raise ValueError(f"{colour!r} is not a six-digit hex colour like '#1f6feb'")
    shades = {
        name: _mix(colour, towards, amount)
        for name, (towards, amount) in _SHADE_MIX.items()
    }
    return gr.themes.Color(name=colour, **shades)


def build_theme() -> gr.themes.Base:
    """Build the Space's theme from the ``THEME_*`` settings in ``config.py``.

    Returns
    -------
    gr.themes.Base
        The theme, to pass to ``launch()``.

    """
    kwargs: dict[str, Any] = {}
    if THEME_FONT:
        kwargs["font"] = [
            gr.themes.GoogleFont(THEME_FONT),
            "ui-sans-serif",
            "sans-serif",
        ]
    return gr.themes.Ocean(
        primary_hue=palette(THEME_PRIMARY_COLOUR),
        secondary_hue=palette(THEME_SECONDARY_COLOUR),
        neutral_hue=palette(THEME_NEUTRAL_COLOUR),
        text_size=sizes.text_lg,
        **kwargs,
    )
