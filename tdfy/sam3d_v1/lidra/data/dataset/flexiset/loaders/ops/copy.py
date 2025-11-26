from copy import deepcopy

from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class Copy(Base):
    def _load(self, data):
        return deepcopy(data)
