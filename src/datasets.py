"""Reusable image datasets for project CSV manifests."""

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class CsvImageDataset(Dataset[tuple[Tensor, int]]):
    """Load RGB images and binary labels from a CSV with path,label columns."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.samples = pd.read_csv(self.csv_path)
        required_columns = {"path", "label"}
        missing_columns = required_columns.difference(self.samples.columns)
        if missing_columns:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing_columns)}")
        if self.samples.empty:
            raise ValueError(f"{self.csv_path} contains no image samples")

        self.samples = self.samples.loc[:, ["path", "label"]].copy()
        self.samples["label"] = pd.to_numeric(self.samples["label"], errors="raise").astype(int)
        if not self.samples["label"].isin([0, 1]).all():
            raise ValueError(f"{self.csv_path} labels must be 0 or 1")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.samples.iloc[index]
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = self.csv_path.parent / image_path

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, int(row["label"])
