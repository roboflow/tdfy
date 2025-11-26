from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class Identity(Base):
    def _load(self, data):
        return data
