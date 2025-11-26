import torch

from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base
from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.numpy.simple_db import (
    SimpleDB as SimpleNumpyDB,
)


class StructuredLatent(Base):
    def __init__(self):
        super().__init__()
        self.register_default_loader("data", SimpleNumpyDB())

    def _load(self, data):
        assert "mean" in data, "Missing 'mean' in loaded data"
        assert "logvar" in data, "Missing 'logvar' in loaded data"

        mean = torch.from_numpy(data["mean"])
        mean = mean.reshape(8, -1)
        mean = mean.transpose(0, 1)

        logvar = torch.from_numpy(data["logvar"])
        logvar = logvar.reshape(8, -1)
        logvar = logvar.transpose(0, 1)

        return mean, logvar
