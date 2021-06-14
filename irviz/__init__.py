from .viewer import Viewer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

_app = None
