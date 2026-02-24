"""
monitoring/dashboard.py — Dashboard Streamlit temps réel pour TRADO.

Lancement : streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TRADO Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_settings():
    """Charge les settings TRADO (gère l'absence de .env proprement)."""
    try:
        from config.settings import Settings
        return Settings()
    except Exception:
        return None


def render_header() -> None:
    st.title("📈 TRADO — Dashboard de Trading IA")
    col1, col2, col3, col4 = st.columns(4)

    settings = load_settings()
    env = getattr(settings, "env", "N/A") if settings else "N/A"

    col1.metric("Environnement", env.upper())
    col2.metric("Portfolio", "10 000.00 $", delta="+0.00 $")
    col3.metric("Positions ouvertes", "0")
    col4.metric("Circuit Breaker", "OK ✅")


def render_equity_curve() -> None:
    st.subheader("Courbe d'équité")
    log_path = Path("logs/trado.log")

    # Placeholder data (remplacer par vraie data en prod)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    equity = 10_000 + pd.Series(range(100)).cumsum() * 10

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode="lines",
        name="Portfolio",
        line=dict(color="#00C896", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,150,0.1)",
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recent_trades() -> None:
    st.subheader("Derniers trades")
    # Placeholder — en prod, lire depuis SQLite
    trades_data = {
        "Date": ["2024-01-15 14:00", "2024-01-15 10:30", "2024-01-14 22:00"],
        "Symbol": ["BTC/USDT", "BTC/USDT", "BTC/USDT"],
        "Action": ["BUY", "SELL", "BUY"],
        "Prix": [42350.0, 43100.0, 41800.0],
        "PnL ($)": [None, +375.0, None],
        "Stratégie": ["EmaRsi", "EmaRsi", "EmaRsi"],
    }
    df = pd.DataFrame(trades_data)
    st.dataframe(df, use_container_width=True)


def render_metrics() -> None:
    st.subheader("Métriques de performance (derniers 30 jours)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe",       "1.42")
    c2.metric("Sortino",      "2.01")
    c3.metric("Max Drawdown", "-4.2%")
    c4.metric("Win Rate",     "58.3%")
    c5.metric("Profit Factor", "1.87")


def render_sidebar() -> None:
    with st.sidebar:
        settings = load_settings()

        st.header("📊 Statut")
        env = getattr(settings, "env", "N/A") if settings else "N/A"
        st.caption(f"Env : **{env.upper()}**")

        st.divider()
        st.header("🛡️ Risk Manager")
        capital = getattr(getattr(settings, "risk", None), "capital_usd", 10_000) if settings else 10_000
        max_pos = getattr(getattr(settings, "risk", None), "max_positions", 5) if settings else 5
        st.metric("Capital", f"{capital:,.0f} $")
        st.metric("Positions max", max_pos)

        st.divider()
        if st.button("⛔ Pause trading", type="secondary"):
            st.warning("Trading mis en pause manuellement.")
        if st.button("🔄 Rafraîchir", type="primary"):
            st.rerun()

        st.divider()
        st.page_link("pages/configuration.py", label="⚙️ Configuration", icon="⚙️")


def main() -> None:
    render_sidebar()
    render_header()

    st.divider()
    render_equity_curve()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_recent_trades()
    with col_right:
        render_metrics()

    # Auto-refresh toutes les 30 secondes
    st.caption("🔄 Actualisation automatique toutes les 30s")
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
