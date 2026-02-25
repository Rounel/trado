"""
launcher.py — Point d'entrée TRADO.exe pour les utilisateurs non-développeurs.

Compilé avec PyInstaller (--onefile --noconsole).
N'importe AUCUNE dépendance ML : uniquement la bibliothèque standard Python.

Comportement :
  1. Détermine PROJECT_DIR (dossier contenant TRADO.exe)
  2. Vérifie que .venv est installé (.venv/Scripts/streamlit.exe)
     → Absent : affiche une fenêtre tkinter et installe les dépendances
  3. Lance Streamlit en arrière-plan
  4. Attend que http://localhost:8501 réponde (60s max)
  5. Ouvre le navigateur par défaut
  6. Attend la fermeture de Streamlit
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


# ──────────────────────────────────────────────
# Résolution du dossier projet
# ──────────────────────────────────────────────

def _project_dir() -> Path:
    """
    Retourne le dossier racine du projet.
    - Si compilé avec PyInstaller : dossier du .exe
    - Si lancé comme script Python : dossier du script
    """
    if getattr(sys, "frozen", False):
        # PyInstaller : sys.executable = chemin vers TRADO.exe
        return Path(sys.executable).parent
    return Path(__file__).parent.resolve()


# ──────────────────────────────────────────────
# Chemins clés
# ──────────────────────────────────────────────

PROJECT    = _project_dir()
VENV       = PROJECT / ".venv"
PYTHON     = VENV / "Scripts" / "python.exe"
STREAMLIT  = VENV / "Scripts" / "streamlit.exe"
DASHBOARD  = PROJECT / "monitoring" / "dashboard.py"
REQS       = PROJECT / "requirements.txt"
PORT       = 8501


# ──────────────────────────────────────────────
# Installation first-run (fenêtre tkinter)
# ──────────────────────────────────────────────

def _run_first_install() -> bool:
    """
    Crée le venv et installe les dépendances via pip.
    Affiche une fenêtre tkinter avec barre de progression et logs.
    Retourne True si succès, False si l'utilisateur a annulé ou si erreur.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    cancelled = [False]

    root = tk.Tk()
    root.title("TRADO — Installation")
    root.geometry("600x420")
    root.resizable(False, False)
    try:
        root.iconbitmap(str(PROJECT / "assets" / "trado.ico"))
    except Exception:
        pass

    tk.Label(
        root,
        text="Installation de TRADO",
        font=("Segoe UI", 14, "bold"),
    ).pack(pady=(20, 4))

    tk.Label(
        root,
        text="Premiere ouverture — installation des composants (~5-10 min selon connexion)",
        font=("Segoe UI", 9),
        fg="#555555",
    ).pack()

    progress = ttk.Progressbar(root, mode="indeterminate", length=540)
    progress.pack(pady=14)
    progress.start(12)

    log_frame = tk.Frame(root)
    log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    scrollbar = tk.Scrollbar(log_frame)
    scrollbar.pack(side="right", fill="y")

    log_box = tk.Text(
        log_frame,
        height=12,
        font=("Consolas", 8),
        bg="#1e1e1e",
        fg="#cccccc",
        wrap="word",
        yscrollcommand=scrollbar.set,
        state="disabled",
    )
    log_box.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=log_box.yview)

    def _append(line: str) -> None:
        log_box.config(state="normal")
        log_box.insert("end", line + "\n")
        log_box.see("end")
        log_box.config(state="disabled")
        root.update_idletasks()

    def _cancel():
        cancelled[0] = True
        root.destroy()

    tk.Button(root, text="Annuler", command=_cancel, width=12).pack(pady=(0, 14))

    def _install():
        try:
            # 1. Créer le venv
            _append("Creation de l'environnement virtuel...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV)],
                cwd=str(PROJECT), check=True,
                capture_output=True,
            )
            _append("Environnement cree.")

            # 2. Mettre à jour pip
            _append("Mise a jour de pip...")
            subprocess.run(
                [str(PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
                cwd=str(PROJECT), check=True,
                capture_output=True,
            )

            # 3. Installer les dépendances (streaming de la sortie)
            _append(f"Installation des dependances depuis {REQS.name}...")
            _append("(cela peut prendre plusieurs minutes...)\n")

            proc = subprocess.Popen(
                [
                    str(PYTHON), "-m", "pip", "install",
                    "-r", str(REQS),
                    "--progress-bar", "off",
                ],
                cwd=str(PROJECT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                if cancelled[0]:
                    proc.terminate()
                    return
                line = line.strip()
                if line:
                    _append(line)

            proc.wait()
            if proc.returncode != 0:
                _append("\nERREUR lors de l'installation.")
                messagebox.showerror(
                    "Erreur",
                    "L'installation a echoue. Verifiez votre connexion internet et relancez TRADO.exe.",
                )
                root.destroy()
                return

            _append("\nInstallation terminee !")
            progress.stop()
            root.destroy()

        except Exception as exc:
            _append(f"\nErreur inattendue : {exc}")
            messagebox.showerror("Erreur", str(exc))
            root.destroy()

    # Lance l'installation dans un thread pour ne pas bloquer le mainloop
    import threading
    threading.Thread(target=_install, daemon=True).start()

    root.mainloop()
    return not cancelled[0] and STREAMLIT.exists()


# ──────────────────────────────────────────────
# Lancement de Streamlit
# ──────────────────────────────────────────────

def _wait_for_server(url: str, timeout: int = 60) -> bool:
    """Attend que le serveur Streamlit réponde. Retourne True si prêt."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _launch_dashboard() -> int:
    """Démarre Streamlit et ouvre le navigateur. Retourne le code de sortie."""
    url = f"http://localhost:{PORT}"

    proc = subprocess.Popen(
        [
            str(STREAMLIT), "run",
            str(DASHBOARD),
            f"--server.port={PORT}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.enableCORS=false",
        ],
        cwd=str(PROJECT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Attendre que le serveur soit prêt avant d'ouvrir le navigateur
    if _wait_for_server(url, timeout=60):
        webbrowser.open(url)
    else:
        # Ouvrir quand même — peut prendre un peu plus de temps
        webbrowser.open(url)

    return proc.wait()


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────

def main() -> None:
    os.chdir(PROJECT)

    # First-run : installer les dépendances si le venv est absent
    if not STREAMLIT.exists():
        if not REQS.exists():
            import tkinter.messagebox as mb
            mb.showerror(
                "TRADO",
                f"Fichier requirements.txt introuvable dans :\n{PROJECT}\n\n"
                "Veuillez reinstaller TRADO.",
            )
            return

        success = _run_first_install()
        if not success:
            return

    # Lancer le dashboard
    sys.exit(_launch_dashboard())


if __name__ == "__main__":
    main()
