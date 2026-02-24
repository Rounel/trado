"""
TRADO — Entrypoint CLI

Usage:
  python main.py run [--broker binance|alpaca|mt5] [--strategy ema_rsi|ai]
                     [--symbols SYM1 SYM2 ...] [--timeframe 1m|5m|15m|30m|1h|4h|1d]
                     [--env paper|live]

  python main.py train [--symbol BTC/USDT] [--timeframe 1h]
                       [--start 2022-01-01] [--end 2024-12-31]
                       [--model tft|rl|all] [--output models/saved]

  python main.py backtest --strategy ema_rsi --symbol BTC/USDT
  python main.py dashboard
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def cli_run() -> None:
    """Lance le moteur de trading en temps réel."""
    parser = _base_parser("run")
    parser.add_argument("--env",       choices=["paper", "live"], default="paper")
    parser.add_argument("--strategy",  choices=["ema_rsi", "ai"], default="ai")
    parser.add_argument("--broker",    choices=["binance", "alpaca", "mt5"], default="binance")
    parser.add_argument("--symbols",   nargs="+", default=None,
                        metavar="SYMBOL",
                        help="Symboles à trader (ex: BTC/USDT ETH/USDT ou EURUSD GBPUSD)")
    parser.add_argument("--timeframe", default="1h",
                        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Timeframe des bougies (défaut: 1h)")
    args = parser.parse_args()

    from config.settings import Settings
    settings = Settings()

    symbols_display = args.symbols or "(défaut broker)"
    print(f"[TRADO] Démarrage — env={args.env}  strategy={args.strategy}")
    print(f"[TRADO] Broker={args.broker}  symbols={symbols_display}  timeframe={args.timeframe}")
    if args.broker == "mt5":
        print(f"[TRADO] MT5 — server={settings.mt5.server}  login={settings.mt5.login}")
    elif args.broker == "binance":
        print(f"[TRADO] Binance testnet={settings.binance.testnet}")

    from core.engine import TradoEngine
    engine = TradoEngine(
        settings=settings,
        strategy_name=args.strategy,
        broker=args.broker,
        symbols=args.symbols,
        timeframe=args.timeframe,
    )
    asyncio.run(engine.start())


def cli_backtest() -> None:
    """Lance un backtest VectorBT sur données historiques."""
    parser = _base_parser("backtest")
    parser.add_argument("--strategy", required=True, choices=["ema_rsi", "ai"])
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    from config.settings import Settings
    settings = Settings()

    print(f"[TRADO] Backtest — strategy={args.strategy}  symbol={args.symbol}")
    print(f"[TRADO] Période : {args.start} → {args.end}  timeframe={args.timeframe}")

    from backtest.engine import BacktestEngine
    bt = BacktestEngine(settings=settings)
    report = bt.run(
        strategy_name=args.strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
    )
    print(report.summary())


def cli_train() -> None:
    """Entraîne les modèles IA (TFT et/ou RL Agent) sur données historiques."""
    parser = _base_parser("train")
    parser.add_argument("--symbol",    default="BTC/USDT",     help="Symbole (ex: BTC/USDT, EURUSD)")
    parser.add_argument("--timeframe", default="1h",            choices=["1m","5m","15m","30m","1h","4h","1d"])
    parser.add_argument("--start",     default="2022-01-01",    help="Début données historiques")
    parser.add_argument("--end",       default="2024-12-31",    help="Fin données historiques")
    parser.add_argument("--model",     default="all",           choices=["tft", "rl", "all"])
    parser.add_argument("--output",    default="models/saved",  help="Dossier de sauvegarde des modèles")
    parser.add_argument("--rl-steps",  type=int, default=500_000, help="Timesteps d'entraînement PPO")
    parser.add_argument("--no-wfo",   action="store_true", help="Désactiver le Walk-Forward (plus rapide)")
    parser.add_argument("--no-cv",    action="store_true", help="Désactiver la cross-validation")
    parser.add_argument("--cv-splits",type=int, default=5, help="Nombre de plis cross-validation")
    args = parser.parse_args()

    from config.settings import Settings
    settings = Settings()

    print(f"[TRADO] Entraînement — symbol={args.symbol}  timeframe={args.timeframe}")
    print(f"[TRADO] Période : {args.start} → {args.end}  modèles={args.model}")
    print(f"[TRADO] WFO={not args.no_wfo}  CV={not args.no_cv}  Sauvegarde → {args.output}/")

    from training.pipeline import TrainingPipeline
    pipeline = TrainingPipeline(
        settings=settings,
        output_dir=args.output,
        rl_timesteps=args.rl_steps,
    )
    saved = pipeline.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        models=args.model,
        run_wfo=not args.no_wfo,
        run_cv=not args.no_cv,
        cv_splits=args.cv_splits,
    )
    print("\n[TRADO] Résultats pipeline :")
    for name, val in saved.items():
        print(f"  {name:20s} → {val}")
    if "wfo_robustness" in saved:
        print(f"\n[TRADO] Robustesse WFO : {saved['wfo_robustness']} (cible : ≥ 60%)")
    if "cv_accuracy" in saved:
        print(f"[TRADO] Accuracy CV    : {saved['cv_accuracy']} (cible : ≥ 45%)")


def cli_stress_test() -> None:
    """Stress test sur des crises historiques (COVID, LUNA, FTX, Bear 2018…)."""
    parser = _base_parser("stress-test")
    parser.add_argument("--strategy", choices=["ema_rsi", "ai"], default="ema_rsi")
    parser.add_argument("--symbol",   default="BTC/USDT")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        metavar="SCENARIO",
                        help="IDs scénarios (défaut=tous). Ex: covid_crash ftx_collapse")
    args = parser.parse_args()

    from config.settings import Settings
    settings = Settings()

    from backtest.stress_test import StressTester, STRESS_SCENARIOS
    print(f"[TRADO] Stress Test — strategy={args.strategy}  symbol={args.symbol}")
    print(f"[TRADO] Scénarios disponibles : {', '.join(STRESS_SCENARIOS.keys())}")

    tester = StressTester(settings=settings)
    report = tester.run(
        strategy_name=args.strategy,
        symbol=args.symbol,
        scenarios=args.scenarios,
    )
    print(report.summary())
    print(f"\n[TRADO] Score de résilience global : {report.resilience_score():.0%}")


def cli_dashboard() -> None:
    """Lance le dashboard Streamlit (monitoring + configuration)."""
    import subprocess
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "monitoring/dashboard.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
        ],
        check=True,
    )


def _base_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"trado {command}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


if __name__ == "__main__":
    commands = {
        "run":         cli_run,
        "train":       cli_train,
        "backtest":    cli_backtest,
        "stress-test": cli_stress_test,
        "dashboard":   cli_dashboard,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python main.py <command> [options]")
        print("")
        print("Commands:")
        for cmd, fn in commands.items():
            print(f"  {cmd:12s}  {fn.__doc__.strip().splitlines()[0]}")
        sys.exit(0)

    cmd = sys.argv.pop(1)
    commands[cmd]()
