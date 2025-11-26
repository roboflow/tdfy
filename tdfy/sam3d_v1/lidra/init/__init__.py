from .environment import init_python_path
from .patch import patch_all

init_python_path()
patch_all()

from . import resolvers
