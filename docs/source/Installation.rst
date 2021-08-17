Software Installation
======================

*PyPXR* can simply be installed using PyPi::
	
	pip install pypxr
	
It requires several standard packages as outlined in requirements.txt provided in the GitHub repository.

Recommended Method
------------------

To avoid conflicting dependencies, the developers recommend isntalling *PyPXR* in a fresh Anaconda environment.

Open a shell window and create a new environment::

	conda create -n pypxr python=3.7 numpy scipy pandas matplotlib seaborn
	
Enter the new environment and finish installing packages as needed::

	# Activate environment
	activate pypxr
	
	# Packages required to reduce data from beamline 11.0.1.2
	conda install astropy
	
	# Packages required for refnx fitting
	conda install h5py xlrd pytest pyqt tqdm openpyxl
	pip install uncertainties attrs periodictable corner refnx
	
	# Install pypxr
	pip install pypxr

