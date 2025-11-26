import os
import sys
from omegaconf import DictConfig
from loguru import logger


def init_env_variables(config: DictConfig):
    """Initialize environment variables from config.

    WARNING: This modifies global environment variables.
    Only call if you're using tdfy in isolation.
    """
    # huggingface cache directory
    if "cluster" in config:
        os.environ["HF_HOME"] = os.path.join(config.cluster.path.cache, "hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


_PYTHON_PATH_INITIALIZED = False


def init_python_path():
    """Add external dependencies to sys.path.

    WARNING: This modifies the global sys.path.
    Only call if you're using tdfy in isolation and need external dependencies.
    Set LIDRA_ENABLE_INIT=1 to enable this.
    """
    global _PYTHON_PATH_INITIALIZED

    if _PYTHON_PATH_INITIALIZED:
        logger.debug("Python path already initialized, skipping")
        return

    current_file_path = os.path.join(os.getcwd(), __file__)
    current_dir_path = os.path.dirname(current_file_path)
    dependencies_path = os.path.join(
        current_dir_path,
        "..",
        "..",
        "external",
        "dependencies",
    )
    dependencies_path = os.path.abspath(dependencies_path)

    # Only add if the path exists and isn't already in sys.path
    if os.path.exists(dependencies_path) and dependencies_path not in sys.path:
        logger.debug(f"Adding external dependencies to sys.path: {dependencies_path}")
        sys.path.insert(0, dependencies_path)
        _PYTHON_PATH_INITIALIZED = True
    elif not os.path.exists(dependencies_path):
        logger.debug(f"External dependencies path does not exist: {dependencies_path}")
