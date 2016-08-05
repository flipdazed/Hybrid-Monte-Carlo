from distutils.core import setup
from distutils.extension import Extension

hello_ext = Extension(
    'fixed',
    sources=['fixed.cpp'],
    libraries=['boost_python-mt'],
)

setup(
    name='fixed',
    version='0.1',
    ext_modules=[hello_ext])