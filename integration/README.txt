###########################################################
## Information-Theoretic Statistics of the Emergent Self ##
###########################################################

This is a Cython package, which means that it has to be compiled. I have compiled it and verified that it works on an installation of Linux Mint 20.1 "Ulyssa". If you are on a Debian-based Linux distribution, I am confident that this will work. If you are on an OSX machine and comfortable with the command line, it should be fine (likewise for any other Unix-like distribution). 

If you are on a Windows machine, you are on your own. Godspeed. 

INSTALLING:

The package can be installed by navigating to this directory in a Shell terminal and executing;

``python setup.py install``

The package requires the following dependencies:

Cython
Numpy
Scipy
Networkx 

And a variety of packages that come native to any Python install (Itertools, Copy). 

USING THE PACKAGE 

After compilation, you can import the functions after adding the directory to your path. A minal script might be:

import sys 
sys.path.append("/path/to/integration.pyx")
from integration import local_total_correlation. 

The package assumes that the data will always be in cells x time format, and is of the type np.float64. If you give it *any other data type* it will at best kindly tell you the data type isn't want it expected, and at worst segfault. Python/MATLAB/R users may not be familiar with static typing. If you're unsure, just add a line redefining your data as the correct type, such as:

data = data.astype("np.float64")

DOCUMENTATION 

In the demo.ipynb Jupyter-notebook, you can see all of the functions demonstrated in detail. 

In the integration.pyx file, you can find detailed documentation about each function, as well as references to the relevant literature. 

