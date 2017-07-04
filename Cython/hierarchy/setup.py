from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

ext = Extension("_hierarchy",sources=["_hierarchy.pyx"])

setup(ext_modules=[ext],cmdclass={'build_ext':build_ext})