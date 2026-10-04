from typing import List, Optional
import gzip
import struct
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        """Read gzip-compressed IDX files; return flattened float32 images in [0, 1]."""
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as stream:
            header = stream.read(16)
            if len(header) != 16:
                raise ValueError("Truncated MNIST image header")
            magic, count, rows, cols = struct.unpack(">IIII", header)
            if magic != 2051 or count == 0 or rows == 0 or cols == 0:
                raise ValueError("Invalid MNIST image header")
            pixels = np.frombuffer(stream.read(), dtype=np.uint8)
        if pixels.size != count * rows * cols:
            raise ValueError("MNIST image payload size does not match its header")
        with gzip.open(label_filename, "rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated MNIST label header")
            magic, labels_count = struct.unpack(">II", header)
            labels = np.frombuffer(stream.read(), dtype=np.uint8)
        if magic != 2049 or labels_count != count or labels.size != count:
            raise ValueError("MNIST labels do not match the images")
        self.images = pixels.reshape(count, rows, cols, 1).astype(np.float32) / 255.0
        self.labels = labels.copy()
        self.image_size = rows * cols

    def __getitem__(self, index) -> object:
        images = self.images[index]
        if images.ndim == 3:
            if self.transforms:
                images = self.apply_transforms(images.copy())
            return images.reshape(self.image_size), self.labels[index]
        if self.transforms:
            images = np.stack([self.apply_transforms(image.copy()) for image in images])
        return images.reshape(len(images), self.image_size), self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
