# TRADO — Bot de Trading IA

Bot de trading algorithmique multi-broker combinant analyse technique avancée (PA/SMC/OTE/IFC), intelligence artificielle, filtrage macro-économique et gestion du risque dynamique.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Utilisation](#utilisation)
7. [Stratégies](#stratégies)
8. [Modules d'analyse avancée](#modules-danalyse-avancée)
9. [Entraînement des modèles IA](#entraînement-des-modèles-ia)
10. [Backtesting & Stress Tests](#backtesting--stress-tests)
11. [Dashboard](#dashboard)
12. [Gestion du risque](#gestion-du-risque)
13. [Tests](#tests)
14. [Brokers supportés](#brokers-supportés)

---

## Vue d'ensemble

TRADO est un bot de trading automatisé conçu pour opérer sur plusieurs marchés simultanément (crypto, actions US). Il repose sur quatre couches d'analyse fusionnées dynamiquement selon le régime de marché détecté :

- **Analyse technique** — EMA, RSI, MACD, Bollinger Bands, ATR, ADX + Price Action / SMC / OTE / IFC
- **Intelligence artificielle** — Temporal Fusion Transformer (TFT) + agent par renforcement (PPO) + Meta-Labeling
- **Sentiment & Macro** — xAI Grok, FinBERT, flux de news, calendrier économique
- **Order Book** — imbalance bid/ask en temps réel

Les poids de chaque couche s'adaptent automatiquement au régime de marché (tendance, range, volatilité extrême).

```
W_TECH = 0.35  |  W_AI = 0.40  |  W_SENTIMENT = 0.15  |  W_ORDERBOOK = 0.10
```

---

## Architecture

```
trado/
├── config/settings.py              Configuration centralisée (Pydantic Settings)
│                                   BinanceConfig, AlpacaConfig, GrokConfig, RedisConfig,
│                                   TelegramConfig, RiskConfig, NewsConfig, MacroFilterConfig
├── core/engine.py                  Orchestrateur async — une tâche par symbole
│
├── data/
│   ├── collectors/
│   │   ├── ohlcv.py                Données OHLCV (ccxt.pro)
│   │   ├── orderbook.py            Order Book temps réel
│   │   ├── grok.py                 Sentiment xAI Grok
│   │   ├── onchain.py              Données on-chain
│   │   ├── economic_calendar.py    Calendrier macro-économique (NFP, CPI, FOMC…)
│   │   └── news_feed.py            Flux de news (RSS / API)
│   └── pipeline/                   Features, normalisation, cache (Redis + SQLite)
│
├── analysis/
│   ├── technical/
│   │   ├── indicators.py           RSI, EMA, MACD, Bollinger Bands, ATR, ADX
│   │   ├── price_action.py         Structure de marché (BOS, CHOCH, tendance)
│   │   ├── smc.py                  Smart Money Concepts (Order Blocks, FVG, Liquidité, Stop Hunt)
│   │   ├── ote.py                  Optimal Trade Entry (retracement Fibonacci 61,8 %–78,6 %)
│   │   └── ifc.py                  Confirmation bougies (Pin Bar, Engulfing, Doji…)
│   ├── sentiment/
│   │   ├── grok_analyzer.py        Analyse Grok / FinBERT
│   │   ├── macro_filter.py         Filtre macro : blackout avant/après événements High
│   │   └── context_builder.py      Contextualisation news + macro par symbole
│   └── signal_fusion.py            Fusion pondérée adaptative (4 couches)
│
├── models/
│   ├── base.py                     Interface abstraite (fit/predict/save/load)
│   ├── lstm.py                     LSTM (PyTorch)
│   ├── tft.py                      Temporal Fusion Transformer (PyTorch)
│   ├── xgboost_model.py            XGBoost classifier
│   ├── rl_agent.py                 Agent PPO (Stable-Baselines3)
│   ├── ensemble.py                 Combinaison des modèles
│   └── meta_labeling.py            Meta-Labeling (filtre probabilité de profit PoP)
│
├── trading/
│   ├── strategies/
│   │   ├── base.py                 Interface BaseStrategy + dataclass Signal
│   │   ├── ema_rsi.py              Stratégie technique pure
│   │   ├── ai_strategy.py          Stratégie IA 4 couches
│   │   └── advanced_strategy.py    Stratégie PA + SMC + OTE + IFC + Meta-Labeling
│   ├── risk/
│   │   ├── manager.py              RiskManager (validation pré-trade)
│   │   ├── sizing.py               PositionSizer (Kelly + ATR)
│   │   └── circuit_breaker.py      Disjoncteur automatique
│   └── executor/
│       ├── binance.py              Exécution ccxt.pro (WebSocket)
│       └── alpaca.py               Exécution Alpaca
│
├── backtest/
│   ├── engine.py                   Simulation barre par barre
│   ├── wfo.py                      Walk-Forward Optimization
│   └── metrics.py                  Sharpe, Sortino, Calmar, Profit Factor
│
├── monitoring/
│   ├── dashboard.py                Dashboard Streamlit temps réel
│   ├── alerts.py                   Alertes Telegram
│   └── logger.py                   Loguru
│
├── tests/
│   ├── unit/                       117 tests (pytest)
│   └── integration/                Pipeline, features, RL env
│
└── main.py                         Entrypoint CLI
```

---

## Prérequis

- **Python** ≥ 3.11
- **uv** (gestionnaire de paquets)

---

## Installation

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd trado

# 2. Installer les dépendances
uv sync

# 3. Copier et remplir le fichier de configuration
cp .env.example .env
```

Pour activer FinBERT (sentiment financier local, ~440 MB) :

```bash
uv sync --extra finbert
```

---

## Configuration

### Via l'interface graphique (recommandé)

```bash
python main.py dashboard
```

Cliquer sur **"Configuration"** dans la sidebar pour accéder aux onglets :
- **Brokers** — clés API Binance, Alpaca
- **Gestion du risque** — capital, circuit breakers, SL/TP
- **Stratégie & Modèles** — stratégie par défaut, chemins des modèles IA
- **Intégrations** — Grok, Telegram, Redis, logs
- **Macro** — blackout avant/après événements, seuil d'impact, multiplicateurs de volatilité

### Via le fichier `.env`

```env
# ── Binance ──────────────────────────────────
BINANCE_API_KEY=votre_cle
BINANCE_SECRET=votre_secret
BINANCE_TESTNET=true          # true = paper trading

# ── Alpaca (Actions US) ──────────────────────
ALPACA_API_KEY=votre_cle
ALPACA_SECRET=votre_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ── Sentiment Grok ───────────────────────────
GROK_API_KEY=votre_cle_grok

# ── Telegram (alertes) ───────────────────────
TELEGRAM_TOKEN=votre_token
TELEGRAM_CHAT_ID=votre_chat_id

# ── Risque ───────────────────────────────────
RISK_CAPITAL_USD=10000.0
RISK_MAX_DRAWDOWN_PCT=10.0
RISK_MAX_DAILY_LOSS_PCT=3.0
RISK_MAX_POSITIONS=5

# ── Filtre macro ─────────────────────────────
MACRO_BLACKOUT_BEFORE_MIN=30
MACRO_BLACKOUT_AFTER_MIN=15
MACRO_MIN_IMPACT=High

# ── News ─────────────────────────────────────
NEWS_MAX_AGE_HOURS=4
NEWS_REFRESH_INTERVAL_MIN=15

# ── App ──────────────────────────────────────
ENV=paper                     # paper | live
LOG_LEVEL=INFO
```

---

## Utilisation

### Lancer le bot

```bash
# Crypto — BTC/USDT (défaut, stratégie ema_rsi)
python main.py run

# Multi-symbole avec stratégie avancée PA/SMC/OTE/IFC
python main.py run --strategy advanced --symbols BTC/USDT ETH/USDT

# Stratégie IA complète (4 couches)
python main.py run --strategy ai --broker binance --symbols BTC/USDT ETH/USDT SOL/USDT

# Actions US via Alpaca
python main.py run --broker alpaca --symbols AAPL TSLA NVDA

# Mode live
python main.py run --env live --strategy advanced
```

### Options disponibles

| Option | Valeurs | Défaut | Description |
|---|---|---|---|
| `--broker` | `binance` `alpaca` | `binance` | Broker d'exécution |
| `--strategy` | `ai` `ema_rsi` `advanced` | `ema_rsi` | Stratégie de trading |
| `--symbols` | liste | selon broker | Instruments à trader |
| `--timeframe` | `1m` `5m` `15m` `30m` `1h` `4h` `1d` | `1h` | Unité de temps |
| `--env` | `paper` `live` | `paper` | Mode paper ou réel |

---

## Stratégies

### `ema_rsi` — Stratégie technique pure

Basée uniquement sur les indicateurs techniques. Recommandée pour débuter ou en l'absence de modèles IA entraînés.

**Signal BUY :** prix > EMA20 > EMA50 > EMA200 + RSI ∈ [45–65] + MACD histogramme > 0 + volume > moyenne 20 bougies

**Signal SELL :** prix < EMA20 < EMA50 < EMA200 + RSI ∈ [35–55] + MACD histogramme < 0 + volume > moyenne

---

### `ai` — Stratégie IA complète (4 couches)

Fusionne quatre sources de signal dont les poids s'adaptent automatiquement au régime de marché :

```
Régime détecté    →  Tech   IA    Sentiment  OB
─────────────────────────────────────────────────
Tendance (ADX>25) →  25%   50%     15%      10%
Range    (ADX<20) →  50%   25%     15%      10%
Volatil  (ATR×1.8)→  30%   20%     35%      15%
Inconnu           →  35%   40%     15%      10%
```

**Score final ≥ +0.20 → BUY | ≤ -0.20 → SELL | entre les deux → HOLD**

> Sans modèles IA entraînés, `ai_score = 0`. Le bot fonctionne sur tech + sentiment + OB (60% de la décision).

---

### `advanced` — Price Action + Smart Money Concepts

Pipeline de génération de signal en 8 étapes :

1. **MarketStructure** — détection de tendance, Break of Structure (BOS), Change of Character (CHOCH)
2. **OrderBlocks** — zones institutionnelles actives (bullish/bearish)
3. **FVGDetector** — Fair Value Gaps (déséquilibres de prix non comblés)
4. **OTEEvaluator** — Optimal Trade Entry (retracement Fibonacci 61,8 %–78,6 % dans un OB)
5. **CandlePatterns (IFC)** — confirmation : Pin Bar, Engulfing, Doji, Morning/Evening Star
6. **LiquidityZones** — zones de concentration de stops (hauts/bas égaux)
7. **StopHuntDetector** — détection de manipulation institutionnelle
8. **MetaLabelingModel** — filtre probabilité de profit (PoP ≥ seuil configurable)

**SL/TP structurels** basés sur les niveaux SMC (pas un multiple d'ATR fixe).

> Le filtre `MacroFilter` s'applique en amont : aucun signal n'est émis pendant les blackouts macro (NFP, CPI, FOMC…).

---

## Modules d'analyse avancée

### Price Action (`analysis/technical/price_action.py`)

Analyse la structure du marché barre par barre :

| Concept | Description |
|---|---|
| **Trend** | Haussier / Baissier / Laterale |
| **BOS** | Break of Structure — confirmation de continuation |
| **CHOCH** | Change of Character — signal potentiel de retournement |
| **Swing High/Low** | Pivots significatifs pour délimiter la structure |

### Smart Money Concepts (`analysis/technical/smc.py`)

| Module | Description |
|---|---|
| `OrderBlockDetector` | Dernière bougie opposée avant un mouvement impulsionnel |
| `FVGDetector` | Gap de prix sur 3 bougies (déséquilibre non comblé) |
| `LiquidityZoneDetector` | Clusters de hauts/bas égaux (zones stop) |
| `StopHuntDetector` | Pics de volume autour des niveaux ronds / swings |

### Optimal Trade Entry (`analysis/technical/ote.py`)

Évalue si le prix se trouve dans la zone OTE (retracement de 61,8 %–78,6 % de Fibonacci) à l'intérieur d'un Order Block actif.

### IFC — Confirmation bougies (`analysis/technical/ifc.py`)

Patterns détectés : Pin Bar, Bullish/Bearish Engulfing, Doji, Morning Star, Evening Star, Hammer, Shooting Star.

### MacroFilter (`analysis/sentiment/macro_filter.py`)

Bloque l'émission de signaux pendant les fenêtres de blackout autour des événements macro à fort impact :

| Paramètre | Défaut | Description |
|---|---|---|
| `blackout_before_min` | 30 min | Fenêtre avant l'événement |
| `blackout_after_min` | 15 min | Fenêtre après l'événement |
| `min_impact` | High | Seuil d'impact minimal |
| `vol_mult_medium` | 1.3× | Multiplicateur ATR (Medium) |
| `vol_mult_high` | 2.0× | Multiplicateur ATR (High) |

### Meta-Labeling (`models/meta_labeling.py`)

Modèle secondaire (XGBoost/RandomForest) qui filtre les signaux de la stratégie primaire. Ne trade que si la probabilité de profit estimée (PoP) dépasse le seuil configuré, réduisant les faux positifs.

---

## Entraînement des modèles IA

### Entraînement standard

```bash
# TFT + RL sur BTC/USDT, 3 ans de données en 1h
python main.py train --symbol BTC/USDT --start 2022-01-01 --end 2024-12-31

# Seulement le TFT (plus rapide)
python main.py train --symbol BTC/USDT --model tft

# Plus de steps pour le RL (meilleure qualité)
python main.py train --symbol BTC/USDT --model rl --rl-steps 1000000
```

### Options d'entraînement

| Option | Défaut | Description |
|---|---|---|
| `--symbol` | `BTC/USDT` | Symbole d'entraînement |
| `--timeframe` | `1h` | Timeframe |
| `--start` | `2022-01-01` | Début des données |
| `--end` | `2024-12-31` | Fin des données |
| `--model` | `all` | `tft` \| `rl` \| `all` |
| `--output` | `models/saved` | Dossier de sauvegarde |
| `--rl-steps` | `500000` | Timesteps PPO |
| `--no-wfo` | — | Désactiver Walk-Forward |
| `--no-cv` | — | Désactiver cross-validation |
| `--cv-splits` | `5` | Nombre de plis CV |

### Ce que le pipeline fait automatiquement

1. Téléchargement des données historiques
2. Calcul des indicateurs (RSI, EMA, MACD, ATR, ADX…)
3. **Walk-Forward Optimization** — mesure la robustesse (alerte si < 60%)
4. **Cross-validation temporelle** (5 plis) — détecte l'overfitting (alerte si accuracy < 45%)
5. Normalisation StandardScaler (sans look-ahead bias)
6. Génération des labels (forward return +3 bougies)
7. Entraînement TFT (50 epochs) + RL Agent (500k timesteps)
8. Sauvegarde dans `models/saved/`

---

## Backtesting & Stress Tests

### Backtest classique

```bash
python main.py backtest --strategy ema_rsi --symbol BTC/USDT \
    --start 2024-01-01 --end 2024-12-31 --timeframe 1h
```

Métriques retournées : Total Return, Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, Win Rate.

### Stress Tests

Évalue la résilience de la stratégie sur 7 crises historiques réelles :

```bash
# Tous les scénarios
python main.py stress-test --strategy ema_rsi

# Scénarios spécifiques
python main.py stress-test --scenarios covid_crash ftx_collapse luna_collapse
```

| ID | Événement | Période |
|---|---|---|
| `covid_crash` | COVID-19 Crash | Fév–Mars 2020 |
| `luna_collapse` | Effondrement LUNA/UST | Mai 2022 |
| `ftx_collapse` | Faillite FTX | Novembre 2022 |
| `crypto_winter_2022` | Bear Market crypto | Juin–Déc 2022 |
| `crypto_bear_2018` | Bear Market 2018 | Jan–Déc 2018 |
| `svb_crisis` | Crise SVB | Mars 2023 |
| `btc_flash_crash_2021` | Flash Crash BTC | Mai 2021 |

Un **score de résilience global** est calculé : proportion de scénarios où la perte reste inférieure à 20%.

---

## Dashboard

```bash
python main.py dashboard
```

Interface Streamlit avec deux pages :

- **Dashboard** — courbe d'équité, derniers trades, métriques (Sharpe, Sortino, Win Rate…)
- **Configuration** — interface graphique pour modifier toutes les variables `.env` sans éditer de fichier

---

## Gestion du risque

Chaque signal passe par le `RiskManager` avant exécution :

| Règle | Valeur par défaut |
|---|---|
| Confiance minimale du signal | ≥ 30% |
| Positions simultanées max | 5 |
| Stop Loss dynamique | `prix − ATR × 2.0` (ou SL structurel SMC) |
| Take Profit dynamique | `prix + ATR × 3.0` (ou TP structurel SMC) |
| **Circuit breaker — drawdown** | > 10% → arrêt total |
| **Circuit breaker — perte journalière** | > 3% → pause du jour |
| **Pertes consécutives** | > 5 → pause automatique |
| **Blackout macro** | avant/après événements High → aucun signal |

Le circuit breaker se réinitialise automatiquement le lendemain (perte journalière) ou manuellement via l'interface.

---

## Tests

```bash
# Tests unitaires (117 tests)
.venv/Scripts/python.exe -m pytest tests/unit/ -v

# Tests d'intégration
.venv/Scripts/python.exe -m pytest tests/integration/ -v

# Couverture complète
.venv/Scripts/python.exe -m pytest tests/ --cov=. -v
```

| Suite | Fichier | Tests |
|---|---|---|
| Indicateurs techniques | `test_indicators.py` | ✓ |
| Gestion du risque | `test_risk.py` | ✓ |
| Stratégies | `test_strategies.py` | ✓ |
| Stratégie avancée (PA/SMC/OTE/IFC) | `test_advanced_strategy.py` | 24 tests |
| Filtre macro-économique | `test_macro_filter.py` | 30 tests |
| Pipeline intégration | `test_pipeline.py` | ✓ |

**Total : 117 tests — durée : ~12 s**

---

## Brokers supportés

| Broker | Marchés | Mode paper | Mode live |
|---|---|---|---|
| **Binance** | Crypto spot | Testnet Binance | API Binance |
| **Alpaca** | Actions US, ETF | Paper API | Live API |

---

## Avertissement

Ce logiciel est fourni à titre éducatif et expérimental. Le trading algorithmique comporte des risques significatifs de perte en capital. Les performances passées ne préjugent pas des performances futures. Utilisez toujours le mode **paper trading** pour valider une stratégie avant tout déploiement en conditions réelles.
