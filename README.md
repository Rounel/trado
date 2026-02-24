# TRADO — Bot de Trading IA

Bot de trading algorithmique multi-broker combinant analyse technique, intelligence artificielle, sentiment de marché et gestion du risque dynamique.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Utilisation](#utilisation)
7. [Stratégies](#stratégies)
8. [Entraînement des modèles IA](#entraînement-des-modèles-ia)
9. [Backtesting & Stress Tests](#backtesting--stress-tests)
10. [Dashboard](#dashboard)
11. [Gestion du risque](#gestion-du-risque)
12. [Brokers supportés](#brokers-supportés)

---

## Vue d'ensemble

TRADO est un bot de trading automatisé conçu pour opérer sur plusieurs marchés simultanément (crypto, forex, actions US). Il repose sur quatre couches d'analyse fusionnées dynamiquement selon le régime de marché détecté :

- **Analyse technique** — EMA, RSI, MACD, Bollinger Bands, ATR, ADX
- **Intelligence artificielle** — Temporal Fusion Transformer (TFT) + agent par renforcement (PPO)
- **Sentiment** — xAI Grok ou FinBERT (spécialisé finance)
- **Order Book** — imbalance bid/ask en temps réel

Les poids de chaque couche s'adaptent automatiquement au régime de marché (tendance, range, volatilité extrême).

---

## Architecture

```
trado/
├── config/settings.py          Configuration centralisée (Pydantic Settings)
├── core/engine.py              Orchestrateur async — une tâche par symbole
│
├── data/
│   ├── collectors/             OHLCV, Order Book, Grok, MT5, On-chain
│   └── pipeline/               Features, normalisation, cache
│
├── analysis/
│   ├── technical/indicators.py RSI, EMA, MACD, BB, ATR, ADX
│   ├── regime/detector.py      Détection régime (trend/range/volatile)
│   ├── sentiment/              Grok API + FinBERT local
│   └── aggregator/signal_fusion.py  Fusion pondérée adaptative
│
├── models/
│   ├── tft/                    Temporal Fusion Transformer (PyTorch)
│   ├── rl_agent/               Agent PPO (Stable-Baselines3)
│   ├── lstm/                   LSTM (PyTorch)
│   ├── xgboost/                XGBoost classifier
│   └── ensemble.py             Combinaison des modèles
│
├── trading/
│   ├── strategies/             ema_rsi, ai_strategy
│   ├── risk/                   RiskManager, CircuitBreaker, PositionSizer
│   └── executor/               Binance, Alpaca, MetaTrader 5
│
├── backtest/
│   ├── engine.py               Simulation barre par barre
│   ├── wfo.py                  Walk-Forward Optimization
│   ├── stress_test.py          7 scénarios de crise historiques
│   └── metrics.py              Sharpe, Sortino, Calmar, Profit Factor
│
├── training/pipeline.py        Pipeline TFT + RL avec WFO et cross-validation
├── monitoring/
│   ├── dashboard.py            Dashboard Streamlit temps réel
│   └── pages/configuration.py Interface de configuration graphique
│
└── main.py                     Entrypoint CLI
```

---

## Prérequis

- **Python** ≥ 3.11
- **uv** (gestionnaire de paquets)
- **Windows** (requis pour MetaTrader 5 uniquement)
- Terminal MetaTrader 5 installé (si broker MT5)

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

Cliquer sur **"Configuration"** dans la sidebar pour accéder aux 4 onglets :
- **Brokers** — clés API Binance, Alpaca, MetaTrader 5
- **Gestion du risque** — capital, circuit breakers, SL/TP
- **Stratégie & Modèles** — stratégie par défaut, chemins des modèles IA
- **Intégrations** — Grok, Telegram, Redis, logs

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

# ── MetaTrader 5 ─────────────────────────────
MT5_LOGIN=12345678
MT5_PASSWORD=votre_mot_de_passe
MT5_SERVER=ICMarkets-Demo
MT5_ENABLED=false

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

# ── App ──────────────────────────────────────
ENV=paper                     # paper | live
LOG_LEVEL=INFO
```

---

## Utilisation

### Lancer le bot

```bash
# Crypto — BTC/USDT (défaut)
python main.py run

# Multi-symbole crypto
python main.py run --broker binance --symbols BTC/USDT ETH/USDT SOL/USDT

# Forex via MetaTrader 5
python main.py run --broker mt5 --symbols EURUSD GBPUSD USDJPY --timeframe 15m

# Actions US via Alpaca
python main.py run --broker alpaca --symbols AAPL TSLA NVDA

# Stratégie technique uniquement (plus simple, moins de dépendances)
python main.py run --strategy ema_rsi
```

### Options disponibles

| Option | Valeurs | Défaut | Description |
|---|---|---|---|
| `--broker` | `binance` `alpaca` `mt5` | `binance` | Broker d'exécution |
| `--strategy` | `ai` `ema_rsi` | `ai` | Stratégie de trading |
| `--symbols` | liste | selon broker | Instruments à trader |
| `--timeframe` | `1m` `5m` `15m` `30m` `1h` `4h` `1d` | `1h` | Unité de temps |
| `--env` | `paper` `live` | `paper` | Mode paper ou réel |

---

## Stratégies

### `ema_rsi` — Stratégie technique pure

Basée uniquement sur les indicateurs techniques. Recommandée pour débuter ou en l'absence de modèles IA entraînés.

**Signal BUY :** prix > EMA20 > EMA50 > EMA200 + RSI ∈ [45–65] + MACD histogramme > 0 + volume > moyenne 20 bougies

**Signal SELL :** prix < EMA20 < EMA50 < EMA200 + RSI ∈ [35–55] + MACD histogramme < 0 + volume > moyenne

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

## Entraînement des modèles IA

L'entraînement utilise des données historiques Binance et intègre automatiquement des mécanismes anti-overfitting.

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

### Charger les modèles dans la stratégie

Via l'interface de configuration (onglet "Stratégie & Modèles"), ou dans le code :

```python
strategy.load_models(
    tft_path="models/saved/tft_BTC_USDT.pt",
    rl_path="models/saved/rl_BTC_USDT",
)
```

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

Ouvre une interface Streamlit dans le navigateur avec deux pages :

- **Dashboard** — courbe d'équité, derniers trades, métriques de performance (Sharpe, Sortino, Win Rate…)
- **Configuration** — interface graphique pour modifier toutes les variables `.env` sans éditer de fichier

---

## Gestion du risque

Chaque signal passe par le `RiskManager` avant exécution :

| Règle | Valeur par défaut |
|---|---|
| Confiance minimale du signal | ≥ 30% |
| Positions simultanées max | 5 |
| Stop Loss dynamique | `prix − ATR × 2.0` |
| Take Profit dynamique | `prix + ATR × 3.0` |
| **Circuit breaker — drawdown** | > 10% → arrêt total |
| **Circuit breaker — perte journalière** | > 3% → pause du jour |
| **Pertes consécutives** | > 5 → pause automatique |

Le circuit breaker se réinitialise automatiquement le lendemain (perte journalière) ou manuellement via l'interface.

---

## Brokers supportés

| Broker | Marchés | Mode paper | Mode live |
|---|---|---|---|
| **Binance** | Crypto spot | Testnet Binance | API Binance |
| **Alpaca** | Actions US, ETF | Paper API | Live API |
| **MetaTrader 5** | Forex, CFD, Indices, Crypto | Compte démo broker | Compte live broker |

> MetaTrader 5 nécessite Windows et le terminal MT5 démarré localement.

---

## Avertissement

Ce logiciel est fourni à titre éducatif et expérimental. Le trading algorithmique comporte des risques significatifs de perte en capital. Les performances passées ne préjugent pas des performances futures. Utilisez toujours le mode **paper trading** pour valider une stratégie avant tout déploiement en conditions réelles.
