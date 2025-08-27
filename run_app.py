# run_app.py
import os, sys
from pathlib import Path
from streamlit.web import cli as stcli


def resource_path(rel_path: str) -> str:
    """Resolve caminho tanto no Python normal quanto no executável PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base = Path(sys._MEIPASS)       # pasta temporária do PyInstaller
    else:
        base = Path(__file__).parent    # pasta do script
    return str((base / rel_path).resolve())

# Garante que o working dir é a pasta do app (funciona mesmo executando de uma rede)
os.chdir(Path(resource_path(".")))

# Ajuste o nome do seu arquivo principal aqui:
APP_FILE = "app.py"

# Flags úteis para ambiente corporativo:
# - --server.port 8501 (mude se precisar)
# - --server.address "localhost"  (só máquina local)
# - --server.fileWatcherType none (evita problemas em pastas de rede)
sys.argv = [
    "streamlit", "run", APP_FILE,
    "--server.address", "localhost",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.fileWatcherType", "none"
]
sys.exit(stcli.main())