"""Initiazlization file for twopoppy"""
__all__ = ['const', 'model', 'args', 'wrapper', 'model_wrapper']
from setuptools_scm import get_version
from .wrapper import model_wrapper
from .args import args
from . import const
from . import model
from . import wrapper

__version__ = get_version(root='..', relative_to=__file__)
