"""KPI card components — reusable metric tiles for dashboard summaries.

``render_kpi_row(cards, mode)`` is the primary entry point.  It renders up to
four metric cards in a single row.  Each card shows a label, a large value, an
optional note, and an optional emoji icon.

No data is fabricated here.  Callers must supply real values derived from the
active dataset.  Passing ``None`` as a value renders a "—" placeholder so the
card is present in the layout without implying a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from config.theme import TOKENS, LIGHT


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class KPICard:
    """Definition for a single metric card.

    Attributes
    ----------
    label:  Short uppercase description, e.g. "TOTAL RECORDS".
    value:  The formatted metric string, e.g. "50,000" or "35.1 %".
            Pass ``None`` to show a dash placeholder.
    note:   Optional sub-label below the value, e.g. "After cleaning".
    icon:   Optional emoji shown to the left of the label, e.g. "📊".
    """
    label: str
    value: str | None
    note: str = ""
    icon: str = ""

    @property
    def display_value(self) -> str:
        return self.value if self.value is not None else "—"


# ── Rendering helpers ──────────────────────────────────────────────────────

def _card_html(card: KPICard, t: dict) -> str:
    """Return the HTML string for a single card using the active token dict."""
    icon_html = (
        f'<span style="font-size:1.3rem; margin-right:6px;">{card.icon}</span>'
        if card.icon
        else ""
    )
    note_html = (
        f'<div style="color:{t["text_secondary"]}; font-size:0.76rem; margin-top:4px;">'
        f"{card.note}</div>"
        if card.note
        else ""
    )
    return f"""
<div style="
    background: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
    box-sizing: border-box;
">
    <div style="
        color: {t['text_secondary']};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: flex;
        align-items: center;
        margin-bottom: 6px;
    ">{icon_html}{card.label}</div>
    <div style="
        color: {t['text_primary']};
        font-family: 'Manrope', sans-serif;
        font-size: 1.72rem;
        font-weight: 800;
        line-height: 1.15;
    ">{card.display_value}</div>
    {note_html}
</div>
"""


def render_kpi_row(
    cards: list[KPICard],
    mode: str = "light",
    max_cols: int = 4,
) -> None:
    """Render a horizontal row of metric cards.

    Parameters
    ----------
    cards:
        List of ``KPICard`` instances.  If more than ``max_cols`` are provided
        they are split into multiple rows of ``max_cols`` each.
    mode:
        Active theme mode — ``"light"`` or ``"dark"``.
    max_cols:
        Maximum cards per row.  Default 4.
    """
    if not cards:
        return

    t = TOKENS.get(mode, LIGHT)

    # Split into rows if there are more cards than max_cols
    for row_start in range(0, len(cards), max_cols):
        row_cards = cards[row_start : row_start + max_cols]
        cols = st.columns(len(row_cards))
        for col, card in zip(cols, row_cards):
            col.markdown(_card_html(card, t), unsafe_allow_html=True)


def render_single_card(card: KPICard, mode: str = "light") -> None:
    """Render a single metric card (for use inside an existing column)."""
    t = TOKENS.get(mode, LIGHT)
    st.markdown(_card_html(card, t), unsafe_allow_html=True)
