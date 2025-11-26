from pytorch3d.io import load_ply

from inference.models.sam3_3d.tdfy.sam3d_v1.lidra.data.dataset.flexiset.loaders.base import Base


class PlyVerticesAndFaces(Base):
    def _load(self, path):
        return load_ply(path)
