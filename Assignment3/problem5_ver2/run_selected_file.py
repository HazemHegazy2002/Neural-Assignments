from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def pick_python_file() -> Path | None:
    try:
        import os

        if os.environ.get("DISPLAY") or sys.platform.startswith("win"):
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            file_path = filedialog.askopenfilename(
                title="Choose a Python file to run",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            )
            root.destroy()

            if file_path:
                return Path(file_path)
    except Exception:
        pass

    try:
        from IPython.display import display
        import ipywidgets as widgets
    except Exception as exc:
        raise RuntimeError(
            "No desktop display is available, and notebook upload widgets are not installed. "
            "Install ipywidgets or run this script on a local desktop Python session."
        ) from exc

    uploader = widgets.FileUpload(accept=".py", multiple=False)
    button = widgets.Button(description="Use uploaded file", button_style="primary")
    output = widgets.Output()
    selected: dict[str, Path | None] = {"path": None}

    def on_click(_):
        if not uploader.value:
            with output:
                print("Upload a .py file first.")
            return

        item = next(iter(uploader.value.values()))
        content = item["content"]
        name = item["metadata"]["name"]
        temp_path = Path(tempfile.gettempdir()) / name
        temp_path.write_bytes(content)
        selected["path"] = temp_path
        with output:
            print(f"Selected: {temp_path}")

    button.on_click(on_click)
    display(uploader, button, output)

    # In notebooks, the user uploads the file and clicks the button.
    # If they don't, the function returns None.
    try:
        input("After uploading the file, press Enter here to continue...")
    except EOFError:
        pass

    return selected["path"]


def run_python_file(file_path: Path) -> int:
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    print(f"Running: {file_path}")
    completed = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=str(file_path.parent),
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print("\n--- stdout ---")
        print(completed.stdout, end="")

    if completed.stderr:
        print("\n--- stderr ---")
        print(completed.stderr, end="")

    print(f"\nExit code: {completed.returncode}")
    return completed.returncode


def main() -> int:
    try:
        chosen_file = pick_python_file()
    except RuntimeError as exc:
        print(exc)
        return 1

    if chosen_file is None:
        print("No file selected.")
        return 0

    if chosen_file.suffix.lower() != ".py":
        answer = input(
            f"Selected file is not a .py file ({chosen_file.name}). Run it with Python anyway? [y/N]: "
        ).strip().lower()
        if answer != "y":
            print("Cancelled.")
            return 0

    return run_python_file(chosen_file)


if __name__ == "__main__":
    raise SystemExit(main())