"""
monitoring/bot_manager.py — Gestion du processus bot de trading.

Permet au dashboard Streamlit de démarrer, arrêter et surveiller le bot
sans que l'utilisateur ait besoin d'un terminal.

Persistance : un fichier PID (data/cache/bot.pid) permet de retrouver
le processus entre les rechargements de page Streamlit.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


_PID_FILE  = Path("data/cache/bot.pid")
_LOG_FILE  = Path("logs/trado.log")
_MAIN_PY   = Path("main.py")


class BotManager:
    """Contrôle le sous-processus `python main.py run ...` depuis le dashboard."""

    # ------------------------------------------------------------------
    # Démarrage
    # ------------------------------------------------------------------

    def start(
        self,
        strategy:  str = "ema_rsi",
        broker:    str = "binance",
        symbols:   list[str] | None = None,
        timeframe: str = "1h",
        env:       str = "paper",
    ) -> bool:
        """
        Lance le bot en arrière-plan.
        Retourne True si le démarrage a réussi, False si le bot tourne déjà.
        """
        if self.is_running():
            return False

        cmd = [
            sys.executable, str(_MAIN_PY), "run",
            "--strategy",  strategy,
            "--broker",    broker,
            "--timeframe", timeframe,
            "--env",       env,
        ]
        if symbols:
            cmd += ["--symbols"] + symbols

        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Ouvre le log en append pour que les sorties s'y accumulent
        log_fd = open(_LOG_FILE, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=log_fd,
            # Sur Windows, CREATE_NEW_PROCESS_GROUP permet un Ctrl+C propre
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            cwd=str(Path(_MAIN_PY).parent.resolve()),
        )

        _PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    # Arrêt
    # ------------------------------------------------------------------

    def stop(self) -> bool:
        """
        Arrête le bot proprement.
        Retourne True si le processus a été stoppé, False si rien ne tournait.
        """
        pid = self._read_pid()
        if pid is None:
            return False

        try:
            import psutil
            proc = psutil.Process(pid)
            # SIGBREAK sur Windows ≈ KeyboardInterrupt pour asyncio
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            # Tuer de force si le signal n'a pas suffi
            try:
                import psutil
                psutil.Process(pid).kill()
            except Exception:
                pass

        _PID_FILE.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # Statut
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Retourne True si le bot tourne (processus PID vivant)."""
        pid = self._read_pid()
        if pid is None:
            return False

        try:
            import psutil
            proc = psutil.Process(pid)
            # Vérifie que c'est bien un processus Python (pas un PID recyclé)
            name = proc.name().lower()
            return proc.is_running() and ("python" in name or "trado" in name)
        except Exception:
            _PID_FILE.unlink(missing_ok=True)
            return False

    def status(self) -> dict:
        """Retourne un dict de statut pour le dashboard."""
        running = self.is_running()
        return {
            "running": running,
            "label":   "En cours" if running else "Arrete",
            "pid":     self._read_pid() if running else None,
        }

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_logs(self, n: int = 50) -> list[str]:
        """Retourne les `n` dernières lignes du fichier de log."""
        if not _LOG_FILE.exists():
            return ["(aucun log disponible)"]
        try:
            with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [l.rstrip() for l in lines[-n:]]
        except Exception as exc:
            return [f"Erreur lecture log : {exc}"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_pid(self) -> int | None:
        """Lit le PID depuis le fichier, retourne None si absent ou invalide."""
        try:
            return int(_PID_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            return None
