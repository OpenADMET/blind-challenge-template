"""Main Gradio app for the blind challenge.

This is the main entry point for the Gradio web application for the challenge. It sets
up the user interface, including tabs each section. The app dynamically adjusts its
content based on the current phase of the challenge which is defined in config.py.
"""

from pathlib import Path

import gradio as gr
from config import (
    ABOUT_MD,
    CHALLENGE_ANNOUCEMENT_LINK,
    CLASSIFICATION_ACTIVE,
    CURRENT_PHASE,
    DATASET_DOWNLOAD_LINK,
    FAQ_MD,
    HEADER_MARKDOWN,
    PAGE_TITLE,
    REGRESSION_ACTIVE,
    STRUCTURE_TRACK_LIVE,
)
from gradio.themes.utils import sizes
from leaderboards import (
    render_final_leaderboards,
    render_interim_leaderboards,
    render_live_leaderboards,
)
from loguru import logger
from submission import render_submission_tab

# Theme and Title setup
with gr.Blocks(
    title=PAGE_TITLE,
    fill_height=False,
    theme=gr.themes.Ocean(text_size=sizes.text_lg),
) as demo:
    ### Header Section
    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown(HEADER_MARKDOWN)
        with gr.Column(scale=2):
            gr.Image(
                value=str(
                    Path(__file__).resolve().parent / "_static" / "challenge_logo.png"
                ),
                show_label=False,
                show_download_button=False,
                width="400px",
            )

    # --- Custom CSS for Tables ---
    gr.HTML("""
            <style>
            #welcome-md table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            #welcome-md th, #welcome-md td {
                padding: 10px;
                border: 1px solid rgba(0,0,0,0.1);
            }
            #welcome-md thead th {
                background: var(--panel-background-fill, #f5f5f7);
            }
            </style>
            """)

    with gr.Row():
        with gr.Column(scale=1):
            if CHALLENGE_ANNOUCEMENT_LINK:
                gr.HTML(f"""
                    <div style="margin: 12px 0 20px 0; text-align: center;">
                        <a href="{CHALLENGE_ANNOUCEMENT_LINK}"
                        target="_blank"
                        style="display: inline-block; padding: 12px 28px; background: #2563eb; color: #fff;
                                font-size: 1.05rem; font-weight: 600; border-radius: 8px; text-decoration: none;
                                box-shadow: 0 2px 8px rgba(37,99,235,0.25);">
                            📄 Challenge Announcement →
                        </a>
                    </div>
                """)
        with gr.Column(scale=1):
            if DATASET_DOWNLOAD_LINK:
                gr.HTML(f"""
                    <div style="margin: 12px 0 20px 0; text-align: center;">
                        <a href="{DATASET_DOWNLOAD_LINK}"
                        target="_blank"
                        style="display: inline-block; padding: 12px 28px; background: #059669; color: #fff;
                                font-size: 1.05rem; font-weight: 600; border-radius: 8px; text-decoration: none;
                                box-shadow: 0 2px 8px rgba(5,150,105,0.25);">
                            🗄️ Training &amp; Test Dataset →
                        </a>
                    </div>
                """)

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
            render_submission_tab(
                phase=CURRENT_PHASE,
                regression_active=REGRESSION_ACTIVE,
                classification_active=CLASSIFICATION_ACTIVE,
                structure_track_live=STRUCTURE_TRACK_LIVE,
            )

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

        # Always display the FAQ tab
        with gr.TabItem("🛠️ FAQ"):
            gr.Markdown(FAQ_MD, elem_id="faq-md")


if __name__ == "__main__":
    logger.info("Starting blind challenge Gradio app...")
    demo.launch()
