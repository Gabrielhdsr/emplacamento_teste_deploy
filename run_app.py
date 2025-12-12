# run_app.py
import os
import sys
from pathlib import Path
from streamlit.web import cli as stcli


def resource_path(rel_path: str) -> Path:
    """Retorna caminho absoluto tanto no Python normal quanto no executável PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)  # PyInstaller
    except AttributeError:
        base_path = Path(__file__).parent  # Execução normal
    return (base_path / rel_path).resolve()


def ensure_app_exists(app_file: str):
    """Valida se o app Streamlit existe antes de tentar executar."""
    if not Path(app_file).exists():
        raise FileNotFoundError(
            f"Arquivo '{app_file}' não encontrado no diretório: {Path.cwd()}"
        )


def main():
    # Garante que o working dir é o diretório real do app
    app_folder = resource_path(".")
    os.chdir(app_folder)

    APP_FILE = "app.py"
    ensure_app_exists(APP_FILE)

    # Flags mais robustas + legíveis
    sys.argv = [
        "streamlit", "run", APP_FILE,
        "--server.address", "localhost",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.fileWatcherType", "watchdog",
        "--theme.base", "dark",
        "--global.developmentMode", "false",
        "--server.runOnSave","true",
    ]

    # Execução isolada
    try:
        sys.exit(stcli.main())
    except Exception as e:
        print("\n❌ ERRO AO INICIAR O STREAMLIT\n")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
