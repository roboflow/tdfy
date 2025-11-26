import os

from tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base
from tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.json.from_file import FromFile


class SimpleDB(Base):
    json_loader = FromFile()

    def __init__(self, extension: str = ".json"):
        super().__init__()
        self.extension = extension

    def _load(self, path, uid):
        filepath = os.path.join(path, f"{uid}{self.extension}")
        return self.json_loader._load(filepath)
