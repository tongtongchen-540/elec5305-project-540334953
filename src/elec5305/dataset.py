from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"slice_file_name", "fold", "classID", "class"}


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: Path
    fold: int
    class_id: int
    class_name: str


class UrbanSound8K:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / "metadata" / "UrbanSound8K.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Expected UrbanSound8K metadata at {metadata_path}. "
                "Pass the extracted dataset root, not the audio folder."
            )
        self.metadata = pd.read_csv(metadata_path)
        missing = REQUIRED_COLUMNS - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    def clips(self, limit: int | None = None) -> list[Clip]:
        rows = self.metadata if limit is None else self.metadata.head(limit)
        clips: list[Clip] = []
        for _, row in rows.iterrows():
            file_name = str(row["slice_file_name"])
            fold = int(row["fold"])
            path = self.root / "audio" / f"fold{fold}" / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing audio referenced by metadata: {path}")
            clips.append(
                Clip(
                    clip_id=file_name.rsplit(".", 1)[0],
                    path=path,
                    fold=fold,
                    class_id=int(row["classID"]),
                    class_name=str(row["class"]),
                )
            )
        return clips
