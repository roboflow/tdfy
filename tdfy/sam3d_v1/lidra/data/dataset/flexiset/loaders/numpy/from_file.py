import os
import numpy as np

from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class FromFile(Base):
    def _load(self, path):
        return np.load(path)
