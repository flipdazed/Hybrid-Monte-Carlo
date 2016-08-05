from distutils.core import setup
from distutils.extension import Extension

hello_ext = Extension(
    'exponential',
    sources=['exponential.cpp'],
    libraries=['boost_python-mt'],
)

setup(
    name='exponential',
    version='0.1',
    ext_modules=[hello_ext])