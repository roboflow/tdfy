import os
import sys

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    from . import init

# Create module aliases for backward compatibility with pickled checkpoints
# When PyTorch unpickles checkpoints, it looks for 'lidra' module paths
# These aliases make 'lidra.*' point to our new location
def _create_module_aliases():
    """Create sys.modules aliases so pickled checkpoints can find lidra modules."""
    # Get all currently loaded modules that start with our new path
    new_prefix = 'tdfy.sam3d_v1.lidra'
    old_prefix = 'lidra'

    for module_name in list(sys.modules.keys()):
        if module_name.startswith(new_prefix):
            # Create alias: tdfy.sam3d_v1.lidra.X -> lidra.X
            old_name = module_name.replace(new_prefix, old_prefix, 1)
            sys.modules[old_name] = sys.modules[module_name]

_create_module_aliases()

# Also set up a meta path finder to handle lazy loading of aliased modules
class LidraModuleAliasFinder:
    """Meta path finder to redirect lidra.* imports to the new location."""

    def find_module(self, fullname, path=None):
        if fullname.startswith('lidra.') or fullname == 'lidra':
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Map lidra.* to tdfy.sam3d_v1.lidra.*
        new_name = fullname.replace('lidra', 'tdfy.sam3d_v1.lidra', 1)

        # Import the actual module
        __import__(new_name)
        module = sys.modules[new_name]

        # Register the alias
        sys.modules[fullname] = module
        return module

# Install the finder if not already installed
if not any(isinstance(finder, LidraModuleAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, LidraModuleAliasFinder())
