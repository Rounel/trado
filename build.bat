@echo off
setlocal EnableDelayedExpansion

echo.
echo ==========================================
echo          TRADO - Build Installer
echo ==========================================
echo.

:: --- [1/5] Verifications ---
echo [1/5] Verification de l'environnement...

if not exist ".venv\Scripts\python.exe" (
    echo ERREUR : .venv non trouve.
    echo Lancez d'abord : uv sync
    pause
    exit /b 1
)

.venv\Scripts\python.exe -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo Installation de PyInstaller...
    .venv\Scripts\pip.exe install "pyinstaller>=6.0" --quiet
    if %errorlevel% neq 0 (
        echo ERREUR : impossible d'installer PyInstaller.
        pause
        exit /b 1
    )
)

echo     OK

:: --- [2/5] Icone ---
echo [2/5] Generation de l'icone...
if not exist "assets" mkdir assets
.venv\Scripts\python.exe scripts\make_icon.py
if %errorlevel% neq 0 (
    echo ERREUR : make_icon.py a echoue.
    pause
    exit /b 1
)

:: --- [3/5] requirements.txt ---
echo [3/5] Generation de requirements.txt...
.venv\Scripts\python.exe scripts\make_requirements.py
if %errorlevel% neq 0 (
    echo ERREUR : make_requirements.py a echoue.
    pause
    exit /b 1
)

:: --- [4/5] PyInstaller ---
echo [4/5] Compilation de TRADO.exe avec PyInstaller...
if exist "build" rmdir /s /q build
.venv\Scripts\pyinstaller.exe trado.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo ERREUR : PyInstaller a echoue. Verifiez les logs ci-dessus.
    pause
    exit /b 1
)

if not exist "dist\TRADO.exe" (
    echo ERREUR : dist\TRADO.exe introuvable apres PyInstaller.
    pause
    exit /b 1
)

for %%F in ("dist\TRADO.exe") do set SIZE_KB=%%~zF
set /a SIZE_MB=%SIZE_KB% / 1048576
echo     TRADO.exe genere dans dist\  (%SIZE_MB% MB)

:: --- [5/5] Inno Setup ---
echo [5/5] Compilation de l'installeur avec Inno Setup...

set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC=C:\Program Files\Inno Setup 6\ISCC.exe
)

if "!ISCC!"=="" (
    echo.
    echo AVERTISSEMENT : Inno Setup 6 non trouve.
    echo Telechargez-le sur : https://jrsoftware.org/isinfo.php
    echo puis relancez build.bat
    echo.
    echo En attendant, TRADO.exe est disponible dans dist\
    echo.
    goto :done
)

"!ISCC!" installer.iss
if %errorlevel% neq 0 (
    echo ERREUR : Inno Setup a echoue.
    pause
    exit /b 1
)

:: --- Resultat ---
:done
echo.
echo ==========================================
echo               Build termine !
echo ==========================================
echo.

if exist "dist\TRADO_setup.exe" (
    for %%F in ("dist\TRADO_setup.exe") do set SETUP_KB=%%~zF
    set /a SETUP_MB=%SETUP_KB% / 1048576
    echo   Installeur : dist\TRADO_setup.exe  (!SETUP_MB! MB)
    echo.
    echo   Envoyez dist\TRADO_setup.exe a votre ami.
    echo   Il double-clique pour installer, puis sur TRADO sur le bureau.
) else (
    echo   Launcher seul : dist\TRADO.exe
)

echo.
pause
