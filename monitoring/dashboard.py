"""
monitoring/dashboard.py — Dashboard Streamlit temps réel pour TRADO.

Sources de données (v2) :
  - data/cache/trado_journal.db (SQLite) — trades fermés + snapshots portfolio
  - config/settings.py — paramètres live (capital, env, …)

Lancement : streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring.bot_manager import BotManager

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TRADO Dashboard",
    page_icon="[chart]",
    layout="wide",
    initial_sidebar_state="expanded",
)

_INITIAL_CAPITAL = 10_000.0   # fallback si pas de settings
_bot = BotManager()


# ──────────────────────────────────────────────
# Chargement des données (cache Streamlit 30 s)
# ──────────────────────────────────────────────

def load_settings():
    try:
        from config.settings import Settings
        return Settings()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_snapshots(limit: int = 500) -> list[dict]:
    from data.pipeline.trade_journal import load_snapshots
    return load_snapshots(limit)


@st.cache_data(ttl=30)
def fetch_trades(limit: int = 50) -> list[dict]:
    from data.pipeline.trade_journal import load_trades
    return load_trades(limit)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _initial_capital(settings) -> float:
    try:
        return float(settings.risk.capital_usd)
    except Exception:
        return _INITIAL_CAPITAL


def _snapshots_to_series(snapshots: list[dict], initial: float) -> tuple[list, list]:
    """Convertit les snapshots en listes (timestamps, valeurs)."""
    if not snapshots:
        return [], []
    ts  = [s["timestamp"] for s in snapshots]
    val = [s["value"]     for s in snapshots]
    return ts, val


def _compute_live_metrics(snapshots: list[dict], trades: list[dict]) -> dict:
    """Calcule Sharpe, Sortino, Max DD, Win Rate, Profit Factor depuis les données réelles."""
    empty = {"sharpe": "N/A", "sortino": "N/A", "max_dd": "N/A",
             "win_rate": "N/A", "profit_factor": "N/A"}

    if len(snapshots) < 10:
        return empty

    try:
        from backtest.metrics import compute_metrics
        values  = pd.Series([s["value"] for s in snapshots])
        pnls    = [t["pnl"] for t in trades] if trades else []
        metrics = compute_metrics(values, pnls)
        return {
            "sharpe":        f"{metrics.sharpe_ratio:.2f}",
            "sortino":       f"{metrics.sortino_ratio:.2f}",
            "max_dd":        f"{metrics.max_drawdown_pct:.1f}%",
            "win_rate":      f"{metrics.win_rate * 100:.1f}%",
            "profit_factor": f"{metrics.profit_factor:.2f}",
        }
    except Exception:
        return empty


# ──────────────────────────────────────────────
# Helpers .env
# ──────────────────────────────────────────────

def _env_str(key: str, default: str = "") -> str:
    """Lit une variable depuis .env sans charger Settings (évite les imports lourds)."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(".env")
        return env.get(key, default) or default
    except Exception:
        return default


# ──────────────────────────────────────────────
# Composants UI
# ──────────────────────────────────────────────

def render_header(snapshots: list[dict], settings) -> None:
    st.title("TRADO — Dashboard de Trading IA")

    initial = _initial_capital(settings)
    env     = getattr(settings, "env", "N/A").upper() if settings else "N/A"

    # Valeur courante et delta depuis le capital initial
    if snapshots:
        current_value = snapshots[-1]["value"]
        positions     = snapshots[-1].get("positions", 0)
        delta_usd     = current_value - initial
        delta_str     = f"{delta_usd:+.2f} $"
    else:
        current_value = initial
        positions     = 0
        delta_str     = "0.00 $"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Environnement",      env)
    col2.metric("Portfolio",          f"{current_value:,.2f} $", delta=delta_str)
    col3.metric("Positions ouvertes", positions)
    col4.metric("Circuit Breaker",    "OK" if snapshots else "N/A")


def render_equity_curve(snapshots: list[dict], settings) -> None:
    st.subheader("Courbe d'équité")
    initial = _initial_capital(settings)

    if snapshots:
        ts, values = _snapshots_to_series(snapshots, initial)
        label      = "Portfolio (live)"
        line_color = "#00C896" if values[-1] >= initial else "#FF4B4B"
    else:
        # Aucune donnée encore — affiche une ligne plate depuis le capital initial
        ts     = pd.date_range("today", periods=2, freq="h").strftime("%Y-%m-%dT%H:%M:%S").tolist()
        values = [initial, initial]
        label  = "Portfolio (en attente de données)"
        line_color = "#888888"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=values,
        mode="lines",
        name=label,
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({','.join(str(int(line_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )

    if not snapshots:
        st.info("Aucun snapshot disponible — démarrez le bot (`python main.py run`) pour alimenter le graphique.", icon="ℹ️")
    st.plotly_chart(fig, use_container_width=True)


def render_recent_trades(trades: list[dict]) -> None:
    st.subheader("Derniers trades")

    if not trades:
        st.info("Aucun trade enregistré.", icon="ℹ️")
        return

    rows = []
    for t in trades:
        pnl   = t.get("pnl", 0)
        rows.append({
            "Date":       t.get("timestamp", "")[:19].replace("T", " "),
            "Symbol":     t.get("symbol", ""),
            "Side":       t.get("side", ""),
            "Entrée":     f"{t.get('entry', 0):.4f}",
            "Sortie":     f"{t.get('exit_price', 0):.4f}",
            "Taille":     f"{t.get('size', 0):.6f}",
            "PnL ($)":    f"{pnl:+.2f}",
            "Stratégie":  t.get("strategy", ""),
        })

    df = pd.DataFrame(rows)

    def _color_pnl(val: str) -> str:
        try:
            return "color: #00C896" if float(val) > 0 else "color: #FF4B4B"
        except ValueError:
            return ""

    st.dataframe(
        df.style.map(_color_pnl, subset=["PnL ($)"]),
        use_container_width=True,
    )


def render_metrics(snapshots: list[dict], trades: list[dict]) -> None:
    st.subheader("Métriques (données réelles)")
    m  = _compute_live_metrics(snapshots, trades)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe",        m["sharpe"])
    c2.metric("Sortino",       m["sortino"])
    c3.metric("Max Drawdown",  m["max_dd"])
    c4.metric("Win Rate",      m["win_rate"])
    c5.metric("Profit Factor", m["profit_factor"])

    if len(snapshots) < 10:
        st.caption(f"Données insuffisantes ({len(snapshots)} snapshots) — les métriques seront disponibles après quelques heures de trading.")


def render_bot_control() -> None:
    """Section démarrage / arrêt du bot, lisible et actionnable par un non-développeur."""
    st.subheader("Controle du bot")

    running = _bot.is_running()

    # Badge statut
    if running:
        st.success("Bot en cours d'execution", icon="🟢")
    else:
        st.error("Bot arrete", icon="🔴")

    col_start, col_stop, col_void = st.columns([1, 1, 3])

    with col_start:
        if st.button("Demarrer", type="primary", disabled=running, use_container_width=True):
            strategy  = _env_str("DEFAULT_STRATEGY",  "ema_rsi")
            broker    = _env_str("DEFAULT_BROKER",    "binance")
            timeframe = _env_str("DEFAULT_TIMEFRAME", "1h")
            env_mode  = _env_str("ENV",               "paper")
            symbols_raw = _env_str("DEFAULT_SYMBOLS", "BTC/USDT")
            symbols   = [s.strip() for s in symbols_raw.split(",") if s.strip()]

            ok = _bot.start(
                strategy=strategy, broker=broker,
                symbols=symbols, timeframe=timeframe, env=env_mode,
            )
            if ok:
                st.success(f"Bot demarre — strategy={strategy} broker={broker} env={env_mode}")
            else:
                st.warning("Le bot tourne deja.")
            time.sleep(1)
            st.rerun()

    with col_stop:
        if st.button("Arreter", type="secondary", disabled=not running, use_container_width=True):
            _bot.stop()
            st.info("Signal d'arret envoye.")
            time.sleep(2)
            st.rerun()

    # Logs récents (expandeur pour ne pas surcharger la page)
    with st.expander("Logs recents (30 dernieres lignes)", expanded=False):
        logs = _bot.get_logs(30)
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.caption("Aucun log disponible.")


def render_sidebar(settings) -> None:
    with st.sidebar:
        st.header("Statut")
        env = getattr(settings, "env", "N/A").upper() if settings else "N/A"
        st.caption(f"Env : **{env}**")

        st.divider()
        st.header("Risk Manager")
        capital = _initial_capital(settings)
        max_pos = getattr(getattr(settings, "risk", None), "max_positions", 5) if settings else 5
        min_rr  = getattr(getattr(settings, "risk", None), "min_rr_ratio",   3.0) if settings else 3.0
        ptsl    = getattr(getattr(settings, "risk", None), "portfolio_trailing_stop_pct", 2.0) if settings else 2.0
        st.metric("Capital",         f"{capital:,.0f} $")
        st.metric("Positions max",   max_pos)
        st.metric("R/R min",         f"{min_rr:.1f}")
        st.metric("Port. Trailing",  f"{ptsl:.1f}%")

        st.divider()
        if st.button("Pause trading", type="secondary"):
            st.warning("Trading mis en pause manuellement.")
        if st.button("Rafraichir", type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        db_path = Path("data/cache/trado_journal.db")
        if db_path.exists():
            size_kb = db_path.stat().st_size // 1024
            st.caption(f"Journal DB : {size_kb} KB")
        else:
            st.caption("Journal DB : non initialisé")

        try:
            st.page_link("pages/configuration.py", label="Configuration", icon="gear")
        except Exception:
            pass


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    settings  = load_settings()
    snapshots = fetch_snapshots(500)
    trades    = fetch_trades(50)

    render_sidebar(settings)
    render_header(snapshots, settings)

    st.divider()
    render_bot_control()

    st.divider()
    render_equity_curve(snapshots, settings)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_recent_trades(trades)
    with col_right:
        render_metrics(snapshots, trades)

    # Auto-refresh toutes les 30 secondes
    st.caption("Actualisation automatique toutes les 30s")
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
