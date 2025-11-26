from loguru import logger

# Track whether patches have been applied to prevent double-patching
_PATCHES_APPLIED = False


def patch_align_device_hook():
    """Patch AlignDevicesHook to work with PyTorch Lightning device mapping.

    WARNING: This modifies a global class from the accelerate library.
    Only call if you're using tdfy in isolation.
    """
    try:
        from accelerate.hooks import AlignDevicesHook

        # Check if already patched
        if hasattr(AlignDevicesHook.pre_forward, '_tdfy_patched'):
            return

        pre_forward = AlignDevicesHook.pre_forward

        def new_pre_forward(self, module, *args, **kwargs):
            try:
                param = next(iter(module.parameters()))
                device = param.device
            except StopIteration:
                device = "cpu"

            self.execution_device = device
            return pre_forward(self, module, *args, **kwargs)

        new_pre_forward._tdfy_patched = True
        AlignDevicesHook.pre_forward = new_pre_forward
    except Exception as e:
        logger.opt(exception=True).warning(f"AlignDevicesHook patching failed: {e}")


def patch_peft():
    """Patch PEFT's dataclass detection.

    WARNING: This modifies the global dataclasses module.
    Only call if you're using tdfy in isolation.
    """
    try:
        import peft
        import dataclasses

        # Check if already patched
        if hasattr(dataclasses.is_dataclass, '_tdfy_patched'):
            return

        not_dataclass_exceptions = (peft.LoraConfig,)
        is_dataclass_fn = dataclasses.is_dataclass

        def patched_is_dataclass(obj):
            return is_dataclass_fn(obj) and not isinstance(obj, not_dataclass_exceptions)

        patched_is_dataclass._tdfy_patched = True
        dataclasses.is_dataclass = patched_is_dataclass
    except Exception as e:
        logger.opt(exception=True).warning(f"peft patching failed: {e}")


def patch_lovely_things():
    """Patch lovely_tensors for better tensor display (optional).

    This is optional and won't affect functionality if it fails.
    """
    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except Exception as e:
        logger.warning(f"error while monkey patching lovely things (optional library): {e}")


def patch_optree():
    """Register OmegaConf DictConfig with optree.

    WARNING: This modifies the global optree registry.
    Only call if you're using tdfy in isolation.
    """
    try:
        import optree
        from omegaconf import DictConfig

        # Check if already registered
        namespace = optree.registry.__GLOBAL_NAMESPACE
        try:
            # Test if already registered
            optree.tree_flatten(DictConfig({}), namespace=namespace)
            return  # Already registered
        except:
            pass

        optree.register_pytree_node(
            DictConfig,
            flatten_func=lambda data: (
                tuple(data.values()),
                tuple(data.keys()),
            ),
            unflatten_func=lambda key, value: dict(zip(key, value)),
            namespace=namespace,
        )
    except Exception as e:
        logger.opt(exception=True).warning(f"optree patching failed: {e}")


def patch_all():
    """Apply all patches.

    WARNING: These patches modify global state and third-party libraries.
    They should ONLY be called when using tdfy in isolation.
    Set LIDRA_ENABLE_INIT=1 to enable these patches.
    """
    global _PATCHES_APPLIED

    if _PATCHES_APPLIED:
        logger.debug("Patches already applied, skipping")
        return

    logger.debug("Applying tdfy global patches (this may affect other libraries)")
    patch_peft()
    patch_align_device_hook()
    patch_lovely_things()
    patch_optree()

    _PATCHES_APPLIED = True
