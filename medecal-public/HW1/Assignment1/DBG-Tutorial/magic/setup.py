# Filename: setup.py

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

MODULES = ['gvmagic']

setup(
    name='ipython-magic',
    version='0.1',

    author="Chris Drake",
    author_email="cjdrake AT gmail DOT com",
    description="IPython magic functions",
    url="https://github.com/cjdrake/ipython-magic",

    py_modules=MODULES,
)

