

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("adabpr_inner",
                 sources=["../code/_adabpr_inner.pyx", "../code/adabpr_inner.c"],
                 include_dirs=[numpy.get_include()])],
)
