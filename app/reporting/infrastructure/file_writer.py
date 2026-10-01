from pathlib import Path


def write_files(directory: str | Path, files: dict[str, str]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
