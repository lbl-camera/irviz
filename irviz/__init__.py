from .viewer import *
from .background_app import *
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
