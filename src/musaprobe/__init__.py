"""musaProbe. Offline analysis does not import PyTorch or a device runtime."""

__version__ = "0.1.0"
__all__ = ["PrecisionDebugger", "ProbeConfig", "__version__"]


def __getattr__(name):
    if name == "PrecisionDebugger":
        from .capture import PrecisionDebugger
        return PrecisionDebugger
    if name == "ProbeConfig":
        from .config import ProbeConfig
        return ProbeConfig
    raise AttributeError(name)
