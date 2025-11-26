import os

# IMPORTANT: Global patching is disabled by default to prevent interference with other packages
# Set LIDRA_ENABLE_INIT=1 to enable initialization (for standalone use only)
# The default is LIDRA_SKIP_INIT for safety
if os.environ.get('LIDRA_ENABLE_INIT') == '1' and not os.environ.get('LIDRA_SKIP_INIT'):
    from . import init

# DISABLED: Module aliasing caused global import interference
# This was creating aliases in sys.modules and sys.meta_path that affected
# all other packages in the Python runtime, not just tdfy.
# If you need checkpoint compatibility, load checkpoints with explicit path mapping instead.
