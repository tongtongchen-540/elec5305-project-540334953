from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Parallel, delayed

from .audio import load_audio
from .config import FeatureConfig
from .dataset import Clip
from .features import extract_feature, feature_manifest


@dataclass
class FeatureSet:
    x: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    clip_ids: np.ndarray
    class_names: np.ndarray

    def save(self, path: str | Path, manifest: dict[str, object]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            folds=self.folds,
            clip_ids=self.clip_ids,
            class_names=self.class_names,
            manifest_json=json.dumps(manifest, sort_keys=True),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["FeatureSet", dict[str, object]]:
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                x=data["x"],
                y=data["y"],
                folds=data["folds"],
                clip_ids=data["clip_ids"],
                class_names=data["class_names"],
            )
            manifest = json.loads(str(data["manifest_json"]))
        return result, manifest


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_feature_set(
    clips: list[Clip],
    config: FeatureConfig,
    transform: Callable[[np.ndarray, Clip], np.ndarray] | None = None,
    n_jobs: int = 1,
) -> FeatureSet:
    def process(clip: Clip) -> np.ndarray:
        audio = load_audio(clip.path, config.sample_rate)
        if transform is not None:
            audio = transform(audio, clip)
        return extract_feature(audio, config)

    vectors = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process)(clip) for clip in clips
    )
    return FeatureSet(
        x=np.vstack(vectors),
        y=np.asarray([clip.class_id for clip in clips], dtype=int),
        folds=np.asarray([clip.fold for clip in clips], dtype=int),
        clip_ids=np.asarray([clip.clip_id for clip in clips]),
        class_names=np.asarray([clip.class_name for clip in clips]),
    )


def cached_feature_set(
    path: str | Path,
    clips: list[Clip],
    config: FeatureConfig,
    transform: Callable[[np.ndarray, Clip], np.ndarray] | None = None,
    n_jobs: int = 1,
    extra_manifest: dict[str, object] | None = None,
    overwrite: bool = False,
) -> FeatureSet:
    path = Path(path)
    expected_manifest = feature_manifest(config)
    if extra_manifest:
        expected_manifest.update(extra_manifest)
    if path.exists() and not overwrite:
        result, stored_manifest = FeatureSet.load(path)
        if all(stored_manifest.get(key) == value for key, value in expected_manifest.items()):
            return result
    result = build_feature_set(clips, config, transform=transform, n_jobs=n_jobs)
    result.save(path, expected_manifest)
    return result
