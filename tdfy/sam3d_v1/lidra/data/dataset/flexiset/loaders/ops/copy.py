from copy import deepcopy

from tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class Copy(Base):
    def _load(self, data):
        return deepcopy(data)
