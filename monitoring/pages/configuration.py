"""
monitoring/pages/configuration.py — Interface de configuration TRADO.

Accessible via : python main.py dashboard → onglet "Configuration" dans la sidebar.
Lit et écrit le fichier .env à la racine du projet.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, set_key

# Chemin vers .env (racine du projet = parent de monitoring/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH      = _PROJECT_ROOT / ".env"
ENV_EXAMPLE   = _PROJECT_ROOT / ".env.example"

st.set_page_config(
    page_title="TRADO — Configuration",
    page_icon="⚙️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _load() -> dict[str, str]:
    """Charge les valeurs depuis .env (ou .env.example si .env absent)."""
    if ENV_PATH.exists():
        return {k: v or "" for k, v in dotenv_values(ENV_PATH).items()}
    if ENV_EXAMPLE.exists():
        return {k: "" for k in dotenv_values(ENV_EXAMPLE)}
    return {}


def _save(updates: dict[str, str]) -> None:
    """Écrit les clés/valeurs dans .env."""
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    for key, value in updates.items():
        set_key(str(ENV_PATH), key, value, quote_mode="never")


def _get(env: dict, key: str, default: str = "") -> str:
    return env.get(key, default) or default


def _status(value: str, label: str = "configuré") -> str:
    return f"✅ {label}" if value and not value.startswith("your_") else "⚠️ non configuré"


# ──────────────────────────────────────────────
# Page principale
# ──────────────────────────────────────────────

st.title("⚙️ Configuration TRADO")
st.caption(f"Fichier : `{ENV_PATH}`")

if not ENV_PATH.exists():
    st.warning("Fichier `.env` introuvable. La configuration sera créée à la première sauvegarde.")

env = _load()

tab_broker, tab_risk, tab_strategy, tab_integrations = st.tabs([
    "🏦 Brokers",
    "🛡️ Gestion du risque",
    "🤖 Stratégie & Modèles IA",
    "🔗 Intégrations",
])

# ══════════════════════════════════════════════
# TAB 1 — Brokers
# ══════════════════════════════════════════════
with tab_broker:
    st.subheader("Brokers actifs")

    broker_col1, broker_col2, broker_col3 = st.columns(3)

    # ── Binance ──────────────────────────────────
    with broker_col1:
        st.markdown("#### 🟡 Binance (Crypto)")
        binance_key    = _get(env, "BINANCE_API_KEY")
        binance_secret = _get(env, "BINANCE_SECRET")
        binance_testnet = _get(env, "BINANCE_TESTNET", "true").lower() == "true"

        st.caption(_status(binance_key, "API Key"))
        b_api = st.text_input("API Key",    value=binance_key,    type="password", key="b_api")
        b_sec = st.text_input("Secret Key", value=binance_secret, type="password", key="b_sec")
        b_test = st.toggle("Paper trading (testnet)", value=binance_testnet, key="b_test")

    # ── Alpaca ───────────────────────────────────
    with broker_col2:
        st.markdown("#### 📊 Alpaca (Actions US)")
        alpaca_key    = _get(env, "ALPACA_API_KEY")
        alpaca_secret = _get(env, "ALPACA_SECRET")
        alpaca_url    = _get(env, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        is_paper      = "paper" in alpaca_url

        st.caption(_status(alpaca_key, "API Key"))
        a_api  = st.text_input("API Key",    value=alpaca_key,    type="password", key="a_api")
        a_sec  = st.text_input("Secret Key", value=alpaca_secret, type="password", key="a_sec")
        a_paper = st.toggle("Paper trading", value=is_paper, key="a_paper")
        a_url  = ("https://paper-api.alpaca.markets" if a_paper
                  else "https://api.alpaca.markets")
        st.caption(f"URL : `{a_url}`")

    # ── MT5 ──────────────────────────────────────
    with broker_col3:
        st.markdown("#### 🖥️ MetaTrader 5 (Forex/CFD)")
        mt5_login    = _get(env, "MT5_LOGIN", "0")
        mt5_password = _get(env, "MT5_PASSWORD")
        mt5_server   = _get(env, "MT5_SERVER")
        mt5_path     = _get(env, "MT5_PATH")
        mt5_enabled  = _get(env, "MT5_ENABLED", "false").lower() == "true"

        st.caption(_status(mt5_server, "Serveur"))
        m_login  = st.text_input("Login (numéro compte)", value=mt5_login,    key="m_login")
        m_pass   = st.text_input("Password",              value=mt5_password, type="password", key="m_pass")
        m_server = st.text_input("Serveur broker",        value=mt5_server,   key="m_server",
                                 placeholder="ex: ICMarkets-Demo")
        m_path   = st.text_input("Chemin terminal64.exe", value=mt5_path,     key="m_path",
                                 placeholder="C:\\Program Files\\MetaTrader 5\\terminal64.exe")
        m_enabled = st.toggle("Activer MT5", value=mt5_enabled, key="m_enabled")

    st.divider()
    if st.button("💾 Sauvegarder les brokers", type="primary", key="save_broker"):
        _save({
            "BINANCE_API_KEY":  b_api,
            "BINANCE_SECRET":   b_sec,
            "BINANCE_TESTNET":  str(b_test).lower(),
            "ALPACA_API_KEY":   a_api,
            "ALPACA_SECRET":    a_sec,
            "ALPACA_BASE_URL":  a_url,
            "MT5_LOGIN":        m_login,
            "MT5_PASSWORD":     m_pass,
            "MT5_SERVER":       m_server,
            "MT5_PATH":         m_path,
            "MT5_ENABLED":      str(m_enabled).lower(),
        })
        st.success("✅ Brokers sauvegardés dans `.env`")
        st.info("Redémarrez le bot pour appliquer les changements.")


# ══════════════════════════════════════════════
# TAB 2 — Gestion du risque
# ══════════════════════════════════════════════
with tab_risk:
    st.subheader("Paramètres de gestion du risque")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 💰 Capital & Positions")
        r_capital = st.number_input(
            "Capital de départ ($)",
            min_value=100.0, max_value=10_000_000.0,
            value=float(_get(env, "RISK_CAPITAL_USD", "10000")),
            step=500.0, format="%.2f",
        )
        r_max_pos = st.number_input(
            "Positions simultanées max",
            min_value=1, max_value=20,
            value=int(_get(env, "RISK_MAX_POSITIONS", "5")),
        )

        st.markdown("#### 📐 Sizing")
        r_kelly = st.slider(
            "Fraction Kelly",
            min_value=0.05, max_value=1.0,
            value=float(_get(env, "RISK_KELLY_FRACTION", "0.25")),
            step=0.05,
            help="0.25 = quart-Kelly (recommandé). Plus élevé = tailles plus grandes.",
        )

    with col_right:
        st.markdown("#### 🔴 Circuit Breakers")
        r_max_dd = st.slider(
            "Drawdown max avant arrêt (%)",
            min_value=1.0, max_value=50.0,
            value=float(_get(env, "RISK_MAX_DRAWDOWN_PCT", "10.0")),
            step=0.5,
            help="Le bot s'arrête si le portefeuille perd plus que ce % depuis son plus haut.",
        )
        r_max_daily = st.slider(
            "Perte journalière max (%)",
            min_value=0.5, max_value=20.0,
            value=float(_get(env, "RISK_MAX_DAILY_LOSS_PCT", "3.0")),
            step=0.5,
        )

        st.markdown("#### 📏 Stop Loss / Take Profit (ATR)")
        r_atr_sl = st.slider(
            "Multiplicateur ATR — Stop Loss",
            min_value=0.5, max_value=5.0,
            value=float(_get(env, "RISK_ATR_SL_MULTIPLIER", "2.0")),
            step=0.5,
            help="SL = prix - ATR × ce multiplicateur",
        )
        r_atr_tp = st.slider(
            "Multiplicateur ATR — Take Profit",
            min_value=1.0, max_value=10.0,
            value=float(_get(env, "RISK_ATR_TP_MULTIPLIER", "3.0")),
            step=0.5,
            help="TP = prix + ATR × ce multiplicateur",
        )

    # Résumé risk:reward
    rr = r_atr_tp / r_atr_sl if r_atr_sl > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Ratio Risk/Reward", f"1 : {rr:.1f}")
    col2.metric("Circuit breaker drawdown", f"{r_max_dd:.1f}%")
    col3.metric("Circuit breaker journalier", f"{r_max_daily:.1f}%")

    st.divider()
    if st.button("💾 Sauvegarder le risque", type="primary", key="save_risk"):
        _save({
            "RISK_CAPITAL_USD":        str(r_capital),
            "RISK_MAX_POSITIONS":      str(r_max_pos),
            "RISK_KELLY_FRACTION":     str(r_kelly),
            "RISK_MAX_DRAWDOWN_PCT":   str(r_max_dd),
            "RISK_MAX_DAILY_LOSS_PCT": str(r_max_daily),
            "RISK_ATR_SL_MULTIPLIER":  str(r_atr_sl),
            "RISK_ATR_TP_MULTIPLIER":  str(r_atr_tp),
        })
        st.success("✅ Paramètres de risque sauvegardés dans `.env`")


# ══════════════════════════════════════════════
# TAB 3 — Stratégie & Modèles IA
# ══════════════════════════════════════════════
with tab_strategy:
    st.subheader("Stratégie de trading")

    col_strat, col_models = st.columns(2)

    with col_strat:
        st.markdown("#### ⚙️ Paramètres par défaut")

        s_broker = st.selectbox(
            "Broker par défaut",
            options=["binance", "alpaca", "mt5"],
            index=["binance", "alpaca", "mt5"].index(
                _get(env, "DEFAULT_BROKER", "binance")
            ),
        )
        s_strategy = st.selectbox(
            "Stratégie par défaut",
            options=["ai", "ema_rsi"],
            index=0 if _get(env, "DEFAULT_STRATEGY", "ai") == "ai" else 1,
        )
        s_timeframe = st.selectbox(
            "Timeframe par défaut",
            options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(
                _get(env, "DEFAULT_TIMEFRAME", "1h")
            ),
        )
        s_symbols_raw = _get(env, "DEFAULT_SYMBOLS", "BTC/USDT")
        s_symbols = st.text_input(
            "Symboles par défaut (séparés par des virgules)",
            value=s_symbols_raw,
            placeholder="BTC/USDT,ETH/USDT,SOL/USDT",
            help="Ces symboles seront tradés si aucun --symbols n'est passé en CLI.",
        )
        s_env = st.selectbox(
            "Environnement",
            options=["paper", "live"],
            index=0 if _get(env, "ENV", "paper") == "paper" else 1,
        )

    with col_models:
        st.markdown("#### 🧠 Modèles IA entraînés")

        saved_dir = _PROJECT_ROOT / "models" / "saved"
        pt_files  = sorted(saved_dir.glob("tft_*.pt")) if saved_dir.exists() else []
        zip_files = sorted(saved_dir.glob("rl_*.zip")) if saved_dir.exists() else []

        # TFT
        tft_options = ["(aucun)"] + [str(f) for f in pt_files]
        tft_current = _get(env, "MODEL_TFT_PATH")
        tft_idx     = tft_options.index(tft_current) if tft_current in tft_options else 0
        m_tft = st.selectbox("Modèle TFT (.pt)", options=tft_options, index=tft_idx)

        # RL
        rl_options  = ["(aucun)"] + [str(f.with_suffix("")) for f in zip_files]
        rl_current  = _get(env, "MODEL_RL_PATH")
        rl_idx      = rl_options.index(rl_current) if rl_current in rl_options else 0
        m_rl = st.selectbox("Modèle RL Agent (.zip)", options=rl_options, index=rl_idx)

        if not pt_files and not zip_files:
            st.info("Aucun modèle entraîné trouvé dans `models/saved/`.\n\n"
                    "Lance `python main.py train` pour entraîner les modèles.")
        else:
            st.success(f"✅ {len(pt_files)} TFT  |  {len(zip_files)} RL trouvés")

        # Poids de fusion
        st.markdown("#### ⚖️ Poids de fusion (AIStrategy)")
        st.caption("Total doit faire 100%")
        w_tech = st.slider("Technique (EMA/RSI/MACD)",  0, 100, 35, key="w_tech")
        w_ai   = st.slider("IA (TFT + RL)",             0, 100, 40, key="w_ai")
        w_sent = st.slider("Sentiment (Grok)",          0, 100, 15, key="w_sent")
        w_ob   = st.slider("Order Book (imbalance)",    0, 100, 10, key="w_ob")
        total  = w_tech + w_ai + w_sent + w_ob
        color  = "green" if total == 100 else "red"
        st.markdown(f"**Total : :{color}[{total}%]**")

    st.divider()
    if st.button("💾 Sauvegarder la stratégie", type="primary", key="save_strat"):
        updates: dict[str, str] = {
            "DEFAULT_BROKER":    s_broker,
            "DEFAULT_STRATEGY":  s_strategy,
            "DEFAULT_TIMEFRAME": s_timeframe,
            "DEFAULT_SYMBOLS":   s_symbols,
            "ENV":               s_env,
        }
        if m_tft != "(aucun)":
            updates["MODEL_TFT_PATH"] = m_tft
        if m_rl != "(aucun)":
            updates["MODEL_RL_PATH"] = m_rl
        if total == 100:
            updates.update({
                "FUSION_W_TECH":      str(w_tech / 100),
                "FUSION_W_AI":        str(w_ai   / 100),
                "FUSION_W_SENTIMENT": str(w_sent / 100),
                "FUSION_W_ORDERBOOK": str(w_ob   / 100),
            })
        _save(updates)
        st.success("✅ Stratégie sauvegardée dans `.env`")


# ══════════════════════════════════════════════
# TAB 4 — Intégrations
# ══════════════════════════════════════════════
with tab_integrations:
    st.subheader("Services externes")

    col_grok, col_notif = st.columns(2)

    with col_grok:
        st.markdown("#### 🤖 xAI Grok (Sentiment)")
        g_key   = _get(env, "GROK_API_KEY")
        g_model = _get(env, "GROK_MODEL", "grok-beta")

        st.caption(_status(g_key, "API Key"))
        g_api = st.text_input("API Key Grok", value=g_key, type="password", key="g_api")
        g_mod = st.selectbox(
            "Modèle",
            options=["grok-beta", "grok-2", "grok-2-mini"],
            index=["grok-beta", "grok-2", "grok-2-mini"].index(g_model)
                  if g_model in ["grok-beta", "grok-2", "grok-2-mini"] else 0,
        )
        st.caption("Sans API Key Grok, le score sentiment = 0 (stratégie toujours fonctionnelle).")

        st.markdown("#### 🗄️ Redis (cache)")
        r_url = st.text_input(
            "URL Redis",
            value=_get(env, "REDIS_URL", "redis://localhost:6379/0"),
            key="r_url",
        )

        st.markdown("#### 📝 Logs")
        log_level = st.selectbox(
            "Niveau de log",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                _get(env, "LOG_LEVEL", "INFO")
            ),
        )
        log_file = st.text_input(
            "Fichier de log",
            value=_get(env, "LOG_FILE", "logs/trado.log"),
        )

    with col_notif:
        st.markdown("#### 📨 Telegram (alertes)")
        t_token   = _get(env, "TELEGRAM_TOKEN")
        t_chat_id = _get(env, "TELEGRAM_CHAT_ID")

        st.caption(_status(t_token, "Bot Token"))
        t_tok  = st.text_input("Bot Token",  value=t_token,   type="password", key="t_tok")
        t_chat = st.text_input("Chat ID",    value=t_chat_id, key="t_chat",
                               placeholder="ex: -100123456789")

        if t_tok and t_chat:
            if st.button("📤 Test notification", key="test_tg"):
                try:
                    import asyncio
                    from telegram import Bot
                    async def _send():
                        bot = Bot(token=t_tok)
                        await bot.send_message(chat_id=t_chat, text="✅ TRADO — Test notification OK")
                    asyncio.run(_send())
                    st.success("Notification envoyée !")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        st.markdown("#### 🌐 Environnement")
        env_mode = st.selectbox(
            "Mode",
            options=["development", "paper", "live"],
            index=["development", "paper", "live"].index(
                _get(env, "ENV", "development")
            ),
        )
        if env_mode == "live":
            st.warning("⚠️ Mode **LIVE** : les ordres seront exécutés avec de l'argent réel !")

    st.divider()
    if st.button("💾 Sauvegarder les intégrations", type="primary", key="save_integrations"):
        _save({
            "GROK_API_KEY":    g_api,
            "GROK_MODEL":      g_mod,
            "REDIS_URL":       r_url,
            "TELEGRAM_TOKEN":  t_tok,
            "TELEGRAM_CHAT_ID": t_chat,
            "LOG_LEVEL":       log_level,
            "LOG_FILE":        log_file,
            "ENV":             env_mode,
        })
        st.success("✅ Intégrations sauvegardées dans `.env`")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Les modifications sont écrites dans `.env` et prises en compte **au prochain démarrage du bot**. "
    "Les clés secrètes ne sont jamais affichées en clair."
)
