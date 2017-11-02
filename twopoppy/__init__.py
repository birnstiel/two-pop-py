"""Initiazlization file for twopoppy"""
__all__ = ['const', 'model', 'args', 'wrapper', 'model_wrapper']
#
# get version
#
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from .wrapper import model_wrapper
from .args import args
from . import const
from . import model
from . import wrapper
