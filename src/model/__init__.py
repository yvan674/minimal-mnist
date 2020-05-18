try:
    from .model import FCNetwork
except ImportError:
    pass
from .numpy_model import NumpyModel

__all__ = ['FCNetwork', 'NumpyModel']
