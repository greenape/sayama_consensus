from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

setup(name = 'sayama_rep',
    include_dirs=[numpy.get_include()],
    ext_modules=[Extension('sayama_rep',
        ['sayama_rep.pyx'])],
    cmdclass = { 'build_ext': build_pyx })