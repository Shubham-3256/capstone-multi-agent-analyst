import os
import shutil
from pathlib import Path

workspace_dir = Path("workspace")

# Files to remove
files_to_remove = [
    workspace_dir / "churn_with_defects.csv",
    workspace_dir / "temp_demo_churn.csv",
    workspace_dir / "data.db",
]

# Folders to clean up contents from
folders_to_clear = [
    workspace_dir / "artifacts",
    workspace_dir / "cleaned",
    workspace_dir / "processed",
    workspace_dir / "uploads",
    workspace_dir / "reports",
    workspace_dir / "plots",
    workspace_dir / "models",
    workspace_dir / "logs",
    workspace_dir / "checkpoints",
    workspace_dir / "cache",
]

for f in files_to_remove:
    if f.exists():
        print(f"Removing file: {f}")
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")

for d in folders_to_clear:
    if d.exists():
        print(f"Clearing folder: {d}")
        for item in d.iterdir():
            if item.name == ".gitkeep":
                continue
            try:
                if item.is_file():
                    os.remove(item)
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Error deleting {item}: {e}")

print("Workspace cleaning complete.")
