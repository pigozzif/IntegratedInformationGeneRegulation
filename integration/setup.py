#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:07:57 2023

@author: thosvarley
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("integration.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
