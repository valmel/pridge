#!/usr/bin/env python
import os, sys
from setuptools import setup

setup(
  name = 'pridge',
  version = '0.1.0',
  description = 'PRIDGE: Ridge regression using PETSc with automatic level of regularization',
  long_description = '--> README.rst',
  url = 'https://github.com/valmel/pridge',
  author = 'Valdemar Melicher',
  author_email = 'Valdemar.Melicher@UAntwerpen.be',
  license = 'MIT',
  classifiers = [
    'DevelopODment Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Machine learning',
    'Topic :: Scientific/Engineering :: Optimization',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
  ],
  keywords = 'Least squares, Linear regression, Tikhonov regularization, hyperparameter tunning, PETSc',
  packages = ['pridge'],
  install_requires = ['numpy', 'scipy', 'mpi4py', 'petsc4py'],
  requires = ['numpy', 'scipy', 'mpi4py', 'petsc4py'],
  package_data = {
    'examples': ['examples/runPETScLSR.py', 'examples/runNumpyLSR.py'],
    'tests': ['tests/testNumpyLSR.py'],
  },
)