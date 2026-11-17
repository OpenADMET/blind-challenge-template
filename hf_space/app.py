"""Main Gradio app for the blind challenge.

This is the main entry point for the Gradio web application for the challenge. It sets
up the user interface, including tabs each section. The app dynamically adjusts its
content based on the current phase of the challenge which is defined in config.py.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import gradio as gr
from banners import model_report_banner, status_text
from config import (
    ABOUT_MD,
    CHALLENGE_ANNOUCEMENT_LINK,
    CURRENT_PHASE,
    DATASET_DOWNLOAD_LINK,
    DISCORD_LINK,
    FAQ_MD,
    FINAL_LEADERBOARD_DEADLINE,
    HEAD_TO_HEAD_ENABLED,
    HEADER_MARKDOWN,
    INTERIM_LEADERBOARD_DEADLINE,
    LOGO_PATH,
    MODEL_REPORT_REMINDER_DAYS,
    PAGE_TITLE,
    REQUIRE_MODEL_REPORT,
    SHOW_TEMPLATE_CREDIT,
    TEMPLATE_CREDIT_MD,
)
from head_to_head import render_head_to_head
from leaderboards import (
    render_final_leaderboards,
    render_interim_leaderboards,
    render_live_leaderboards,
)
from loguru import logger
from submission import render_submission_tab
from theme import build_theme

HERE = Path(__file__).resolve().parent
LOGO = HERE / LOGO_PATH
CSS = HERE / "style.css"
BANNER_REFRESH_SECONDS = 300


def _banners() -> tuple[str, dict]:
    """Build the status line and model report reminder for the current time."""
    now = datetime.now(timezone.utc)
    status = status_text(
        CURRENT_PHASE, now, INTERIM_LEADERBOARD_DEADLINE, FINAL_LEADERBOARD_DEADLINE
    )
    reminder = model_report_banner(
        CURRENT_PHASE,
        now,
        FINAL_LEADERBOARD_DEADLINE,
        REQUIRE_MODEL_REPORT,
        MODEL_REPORT_REMINDER_DAYS,
    )
    return status, gr.update(value=reminder, visible=bool(reminder))


with gr.Blocks(title=PAGE_TITLE, fill_height=False) as demo:
    with gr.Row(elem_classes="challenge-header"):
        with gr.Column(scale=7):
            gr.Markdown(HEADER_MARKDOWN)
        # The template ships without a logo; the header uses the full width until
        # one is added at LOGO_PATH.
        if LOGO.exists():
            with gr.Column(scale=2, min_width=160):
                gr.Image(
                    value=str(LOGO),
                    show_label=False,
                    buttons=[],
                    container=False,
                    width="400px",
                )

    links: list[tuple[str, str | bool, Literal["primary", "secondary"]]] = [
        ("📄 Challenge Announcement", CHALLENGE_ANNOUCEMENT_LINK, "primary"),
        ("🗄️ Training & Test Dataset", DATASET_DOWNLOAD_LINK, "secondary"),
        ("💬 Discord", DISCORD_LINK, "secondary"),
    ]
    with gr.Row(elem_classes="link-buttons"):
        for label, url, variant in links:
            if url:
                gr.Button(
                    label,
                    link=str(url),
                    link_target="_blank",
                    variant=variant,
                    scale=1,
                )

    status_line = gr.Markdown(elem_classes="status-line")
    reminder_banner = gr.HTML(visible=False)
    # Rebuilt on every page load and every few minutes, so they follow the date
    # without restarting the Space.
    gr.on(
        triggers=[demo.load, gr.Timer(value=BANNER_REFRESH_SECONDS).tick],
        fn=_banners,
        outputs=[status_line, reminder_banner],
    )

    with gr.Tabs(elem_classes="tab-buttons"):
        # Possible phases:
        # 0: Pre-challenge
        # 1: Phase 1 (only live leaderboard)
        # 2: Phase 2 (live leaderboard + static interim leaderboard)
        # 3: Challenge closed, final leaderboard not yet released
        # 4: Challenge closed, final leaderboard released

        # Always display the About tab
        with gr.TabItem("📖 About"):
            gr.Markdown(ABOUT_MD, elem_id="welcome-md")

        # Always display the submission tab
        # During phase 0 (pre-challenge), there will be placeholder text
        # After phase 3 (end of challenge), there will be placeholder text
        with gr.TabItem("✉️ Submit"):
            render_submission_tab(phase=CURRENT_PHASE)

        # Display live leaderboard up until end of challenge
        # During phase 0 (pre-challenge), there will be placeholder markdown
        if CURRENT_PHASE in [0, 1, 2]:
            with gr.TabItem("🏎️ Live Leaderboard"):
                render_live_leaderboards(phase=CURRENT_PHASE, demo=demo)

        # Display the interim leaderboard tab from phase 2 onwards
        if CURRENT_PHASE in [2, 3, 4]:
            with gr.TabItem("⏱️ Interim Leaderboard"):
                render_interim_leaderboards()

        # Display the final leaderboard tab from phase 3 (end of challenge) onwards
        # Before the leaderboard is ready, there will be placeholder markdown (phase 3)
        if CURRENT_PHASE in [3, 4]:
            with gr.TabItem("🏁 Final Leaderboard"):
                render_final_leaderboards(phase=CURRENT_PHASE)

        # Compare two final-leaderboard entrants, once the final leaderboard is out
        if CURRENT_PHASE == 4 and HEAD_TO_HEAD_ENABLED:
            with gr.TabItem("⚔️ Head-to-Head"):
                render_head_to_head()

        # Always display the FAQ tab
        with gr.TabItem("🛠️ FAQ"):
            gr.Markdown(FAQ_MD, elem_id="faq-md")

    if SHOW_TEMPLATE_CREDIT:
        gr.Markdown(TEMPLATE_CREDIT_MD, elem_classes="template-credit")


if __name__ == "__main__":
    logger.info("Starting blind challenge Gradio app...")
    # No "Use via API" link: submissions need a Hugging Face login, which API calls
    # don't carry, so they're refused anyway.
    demo.launch(
        theme=build_theme(), css_paths=[CSS], footer_links=["gradio", "settings"]
    )
