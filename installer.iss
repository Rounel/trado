; installer.iss — Script Inno Setup 6 pour TRADO
;
; Prérequis : Inno Setup 6 (https://jrsoftware.org/isinfo.php)
; Pour construire : build.bat  ou  iscc installer.iss
;
; Ce que fait l'installeur :
;   1. Copie TRADO.exe + le code source dans %LOCALAPPDATA%\TRADO\
;   2. Copie requirements.txt (utilisé par TRADO.exe au 1er lancement)
;   3. Crée un raccourci sur le Bureau
;   4. Crée une entrée dans Démarrer > Programmes
;   5. Installe un désinstalleur propre
;
; NOTE : les dépendances Python (torch, streamlit, etc.) sont
; téléchargées et installées par TRADO.exe lui-même au premier
; lancement (via pip). L'installeur reste donc léger (~20 MB).

#define MyAppName      "TRADO"
#define MyAppVersion   "0.1.0"
#define MyAppPublisher "TRADO Trading"
#define MyAppURL       "https://github.com/trado"
#define MyAppExeName   "TRADO.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Pas besoin de droits administrateur — installation utilisateur
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=TRADO_setup
SetupIconFile=assets\trado.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Pas de page de sélection de dossier (installation silencieuse possible)
DisableDirPage=no
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le Bureau"; GroupDescription: "Raccourcis :"; Flags: checked

[Files]
; Launcher compilé
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Fichier de dépendances pour le first-run
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Code source (hors .venv, .git, __pycache__, dist, logs)
Source: "config\*";      DestDir: "{app}\config";     Flags: ignoreversion recursesubdirs createallsubdirs
Source: "core\*";        DestDir: "{app}\core";        Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\*";        DestDir: "{app}\data";        Flags: ignoreversion recursesubdirs createallsubdirs excludesubdirs; Excludes: "*.db,*.jsonl,*.gz"
Source: "models\*";      DestDir: "{app}\models";      Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "saved\*"
Source: "analysis\*";    DestDir: "{app}\analysis";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "trading\*";     DestDir: "{app}\trading";     Flags: ignoreversion recursesubdirs createallsubdirs
Source: "backtest\*";    DestDir: "{app}\backtest";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "monitoring\*";  DestDir: "{app}\monitoring";  Flags: ignoreversion recursesubdirs createallsubdirs
Source: "training\*";    DestDir: "{app}\training";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "main.py";       DestDir: "{app}";             Flags: ignoreversion
Source: ".env.example";  DestDir: "{app}";             DestName: ".env.example"; Flags: ignoreversion

; Icône
Source: "assets\trado.ico"; DestDir: "{app}\assets"; Flags: ignoreversion

[Icons]
; Raccourci Bureau
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; \
    IconFilename: "{app}\assets\trado.ico"; \
    Tasks: desktopicon

; Menu Démarrer
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; \
    IconFilename: "{app}\assets\trado.ico"

Name: "{group}\Désinstaller {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
; Option : lancer TRADO immédiatement après l'installation
Filename: "{app}\{#MyAppExeName}"; \
    Description: "Lancer {#MyAppName} maintenant"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Supprimer le venv et les fichiers générés à la désinstallation
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\data\cache"
