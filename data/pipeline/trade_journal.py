"""
data/pipeline/trade_journal.py — Journal persistant des trades et snapshots portfolio.

Deux APIs :
  - Async (pour le TradoEngine en production)    : record_closed_trade(), record_snapshot()
  - Sync  (pour le dashboard Streamlit)          : load_trades(), load_snapshots()

Base de données : data/cache/trado_journal.db (SQLite, séparé du cache OHLCV)
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import aiosqlite
from loguru import logger

DB_PATH = "data/cache/trado_journal.db"

_CREATE_CLOSED_TRADES = """
CREATE TABLE IF NOT EXISTS closed_trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    side       TEXT    NOT NULL,
    entry      REAL    NOT NULL,
    exit_price REAL    NOT NULL,
    size       REAL    NOT NULL,
    pnl        REAL    NOT NULL,
    strategy   TEXT
)"""

_CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    value     REAL NOT NULL,
    positions INTEGER DEFAULT 0
)"""

_CREATE_OPEN_POSITIONS = """
CREATE TABLE IF NOT EXISTS open_positions (
    symbol               TEXT PRIMARY KEY,
    side                 TEXT NOT NULL,
    size                 REAL NOT NULL,
    entry                REAL NOT NULL,
    stop_loss            REAL,
    trailing_stop        REAL,
    trailing_activation  REAL,
    opened_at            TEXT NOT NULL
)"""


def _ensure_db_dir() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────

def init_sync() -> None:
    """Crée les tables si elles n'existent pas (appelé au démarrage du dashboard)."""
    _ensure_db_dir()
    con = sqlite3.connect(DB_PATH)
    con.execute(_CREATE_CLOSED_TRADES)
    con.execute(_CREATE_SNAPSHOTS)
    con.execute(_CREATE_OPEN_POSITIONS)
    con.commit()
    con.close()


async def init_async() -> None:
    """Crée les tables si elles n'existent pas (appelé au démarrage de l'engine)."""
    _ensure_db_dir()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_CLOSED_TRADES)
        await db.execute(_CREATE_SNAPSHOTS)
        await db.execute(_CREATE_OPEN_POSITIONS)
        await db.commit()


# ──────────────────────────────────────────────────────────────
# Écriture async (depuis TradoEngine)
# ──────────────────────────────────────────────────────────────

async def record_closed_trade(
    symbol:     str,
    side:       str,
    entry:      float,
    exit_price: float,
    size:       float,
    pnl:        float,
    strategy:   str | None = None,
) -> None:
    """Enregistre un trade fermé."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO closed_trades
                   (timestamp, symbol, side, entry, exit_price, size, pnl, strategy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol, side,
                    round(entry, 6), round(exit_price, 6),
                    round(size, 8),  round(pnl, 4),
                    strategy,
                ),
            )
            await db.commit()
    except Exception as exc:
        logger.warning(f"TradeJournal.record_closed_trade: {exc}")


async def record_snapshot(value: float, positions: int = 0) -> None:
    """Enregistre un snapshot de la valeur du portefeuille."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO portfolio_snapshots (timestamp, value, positions) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), round(value, 4), positions),
            )
            await db.commit()
    except Exception as exc:
        logger.warning(f"TradeJournal.record_snapshot: {exc}")


# ──────────────────────────────────────────────────────────────
# Positions ouvertes — persistance inter-sessions
# ──────────────────────────────────────────────────────────────

async def save_open_position(
    symbol:              str,
    side:                str,
    size:                float,
    entry:               float,
    stop_loss:           float | None = None,
    trailing_stop:       float | None = None,
    trailing_activation: float | None = None,
) -> None:
    """Insère ou met à jour une position ouverte (upsert par symbole)."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO open_positions
                   (symbol, side, size, entry, stop_loss, trailing_stop,
                    trailing_activation, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol) DO UPDATE SET
                     side=excluded.side, size=excluded.size,
                     entry=excluded.entry, stop_loss=excluded.stop_loss,
                     trailing_stop=excluded.trailing_stop,
                     trailing_activation=excluded.trailing_activation""",
                (
                    symbol, side,
                    round(size, 8), round(entry, 6),
                    round(stop_loss, 6) if stop_loss is not None else None,
                    round(trailing_stop, 6) if trailing_stop is not None else None,
                    round(trailing_activation, 6) if trailing_activation is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
    except Exception as exc:
        logger.warning(f"TradeJournal.save_open_position [{symbol}]: {exc}")


async def delete_open_position(symbol: str) -> None:
    """Supprime une position fermée du journal."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
            await db.commit()
    except Exception as exc:
        logger.warning(f"TradeJournal.delete_open_position [{symbol}]: {exc}")


async def load_open_positions(symbol: str | None = None) -> list[dict]:
    """
    Charge les positions ouvertes depuis le journal.
    Si `symbol` est fourni, ne retourne que celle de ce symbole.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if symbol:
                rows = await db.execute_fetchall(
                    "SELECT * FROM open_positions WHERE symbol = ?", (symbol,)
                )
            else:
                rows = await db.execute_fetchall("SELECT * FROM open_positions")
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"TradeJournal.load_open_positions: {exc}")
        return []


# ──────────────────────────────────────────────────────────────
# Lecture sync (depuis le dashboard Streamlit)
# ──────────────────────────────────────────────────────────────

def load_snapshots(limit: int = 500) -> list[dict]:
    """
    Retourne les `limit` derniers snapshots (ordre chronologique ASC).
    Retourne [] si la base n'existe pas encore.
    """
    try:
        _ensure_db_dir()
        init_sync()
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        return [dict(r) for r in reversed(rows)]
    except Exception as exc:
        logger.debug(f"TradeJournal.load_snapshots: {exc}")
        return []


def load_trades(limit: int = 50) -> list[dict]:
    """
    Retourne les `limit` derniers trades fermés (ordre chronologique DESC).
    Retourne [] si la base n'existe pas encore.
    """
    try:
        _ensure_db_dir()
        init_sync()
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM closed_trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug(f"TradeJournal.load_trades: {exc}")
        return []
