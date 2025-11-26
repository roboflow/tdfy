from tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class Identity(Base):
    def _load(self, data):
        return data
