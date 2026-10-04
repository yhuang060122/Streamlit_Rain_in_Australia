"""Step-gating for the CRISP-DM data flow.

Each analysis page depends on the previous one. If the previous step hasn't been
visited (run) yet, the current page shows an error instead of executing.
"""

from __future__ import annotations

import streamlit as st

# Linear order of the analysis pipeline. Each step depends on the one before it.
STEP_ORDER = [
    "data_understanding",
    "eda",
    "data_preprocessing",
    "feature_engineering",
    "modeling",
    "model_interpretation",
]


def require(prev_step: str) -> None:
    """Block the page if the previous step hasn't been completed yet."""
    if not st.session_state.get(f"step_{prev_step}_done", False):
        st.error(
            f"⚠️ This step depends on **{prev_step.replace('_', ' ').title()}**. "
            f"Please go back and run that step first."
        )
        st.stop()


def mark_done(step: str) -> None:
    """Mark the current step as completed (visited)."""
    st.session_state[f"step_{step}_done"] = True
