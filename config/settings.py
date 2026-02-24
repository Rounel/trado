"""
config/settings.py — Configuration centralisée via Pydantic Settings.
Toutes les valeurs sont chargées depuis les variables d'environnement / fichier .env.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_", extra="ignore")

    api_key: str = Field(default="", description="Binance API key")
    secret: str = Field(default="", description="Binance secret")
    testnet: bool = Field(default=True, description="True = paper trading sur testnet")


class AlpacaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_", extra="ignore")

    api_key: str = Field(default="", description="Alpaca API key")
    secret: str = Field(default="", description="Alpaca secret")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="URL de l'API Alpaca (paper ou live)",
    )


class GrokConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROK_", extra="ignore")

    api_key: str = Field(default="", description="xAI Grok API key")
    base_url: str = Field(default="https://api.x.ai/v1")
    model: str = Field(default="grok-beta")


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    url: str = Field(default="redis://localhost:6379/0")


class TelegramConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", extra="ignore")

    token: str = Field(default="")
    chat_id: str = Field(default="")


class MT5Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MT5_", extra="ignore")

    login: int = Field(default=0, description="Numéro de compte MT5")
    password: str = Field(default="", description="Mot de passe MT5")
    server: str = Field(default="", description="Serveur broker MT5 (ex: ICMarkets-Demo)")
    path: str = Field(default="", description="Chemin vers terminal64.exe (optionnel)")
    timeout: int = Field(default=60_000, description="Timeout connexion en ms")
    enabled: bool = Field(default=False, description="True = utiliser MT5 comme broker actif")


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore")

    max_drawdown_pct: float = Field(default=10.0, description="Drawdown max avant circuit breaker (%)")
    max_daily_loss_pct: float = Field(default=3.0, description="Perte journalière max (%)")
    capital_usd: float = Field(default=10_000.0, description="Capital total")
    max_positions: int = Field(default=5, description="Nombre max de positions ouvertes simultanées")
    kelly_fraction: float = Field(default=0.25, description="Fraction Kelly (0.25 = quart-Kelly)")
    atr_sl_multiplier: float = Field(default=2.0, description="Multiplicateur ATR pour Stop Loss")
    atr_tp_multiplier: float = Field(default=3.0, description="Multiplicateur ATR pour Take Profit")


class Settings(BaseSettings):
    """Settings globaux de l'application TRADO."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = Field(default="development", description="development | paper | live")
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trado.log")

    # Sous-configs imbriquées
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    mt5: MT5Config = Field(default_factory=MT5Config)
    grok: GrokConfig = Field(default_factory=GrokConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
