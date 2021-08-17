Installation
=============

*PyPXR* is currently in ongoing development and does not have an easy installation. This page will help guide you toward the developers prefered method for running it in Jupyter on a new computer.

While not required, it is recommended to install within a conda environment::

	conda create -n pypxr python=3.7 numpy scipy pandas xlrd pytest refnx matplotlib

Enter the newly created environment and finish installing packages through pip::

	activate pypxr #On windows
	pip install uncertainties attrs periodictable
	
Lastly, clone the GitHub directory onto your machine.

To import functions from pypxr, run the following imports at the start of any notebook/python file::
	
	import sys
	sys.path.append("/my/path/to/PyPXR/")
	from pypxr.anisotropic_reflectivity import *
	from pypxr.anisotropic_structure import *
	
This should be enough to run pypxr on your machine. To verify if the install worked, run the following function::

	python pypxr_test.py
	
