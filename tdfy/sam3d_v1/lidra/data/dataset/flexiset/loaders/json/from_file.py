import json

from tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class FromFile(Base):
    def _load(self, path):
        with open(path, "r") as f:
            return json.load(f)
