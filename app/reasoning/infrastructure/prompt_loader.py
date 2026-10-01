from functools import cache
from pathlib import Path

PROMPT_VERSION = "v1"  # bump when wording changes, so cached answers to old prompts are not reused
_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(name: str, version: str = PROMPT_VERSION) -> str:
    """Prompts live in text files (prompts/<version>/<name>.txt), never inline in code."""
    return (_DIR / version / f"{name}.txt").read_text(encoding="utf-8").strip()
