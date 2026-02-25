# trado.spec — Spec PyInstaller pour TRADO.exe
#
# Ne bundle QUE le launcher.py et la stdlib Python.
# Les dépendances ML (torch, ccxt, streamlit...) sont installées
# séparément dans .venv/ lors du premier lancement.
#
# Construire :
#   pyinstaller trado.spec
# ou via :
#   build.bat
#
# Résultat : dist/TRADO.exe (~8-15 MB)

import sys
from pathlib import Path

block_cipher = None

# Dossier racine du projet
PROJECT_ROOT = str(Path(SPECPATH))

a = Analysis(
    ['launcher.py'],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # Inclut l'icône si elle existe
        ('assets/trado.ico', 'assets') if Path('assets/trado.ico').exists() else ('', '.'),
    ],
    hiddenimports=[
        # Modules stdlib utilisés par le launcher
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'threading',
        'urllib.request',
        'webbrowser',
        'subprocess',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclure explicitement les gros paquets ML pour garder le .exe léger
        'torch', 'transformers', 'xgboost', 'sklearn',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'plotly',
        'streamlit', 'ccxt', 'alpaca_trade_api',
        'stable_baselines3', 'gymnasium', 'optuna',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TRADO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                          # UPX compresse le binaire (~30% de gain)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                     # Pas de fenêtre console noire
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/trado.ico' if Path('assets/trado.ico').exists() else None,
    version='version_info.txt' if Path('version_info.txt').exists() else None,
)
