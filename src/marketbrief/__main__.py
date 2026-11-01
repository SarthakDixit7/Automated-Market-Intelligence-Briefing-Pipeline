"""Allow ``python -m marketbrief``."""

from marketbrief.cli import app

if __name__ == "__main__":
    app(prog_name="marketbrief")
