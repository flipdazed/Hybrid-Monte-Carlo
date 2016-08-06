Theoretical Autocorrelation and Integrated Autocorrelations
======


## Making wrapper files
Manually include all functions from respective directories into 
	
	exponential.cpp`
	fixed.cpp`

See [this guide](https://github.com/flipdazed/boost-python-hello-world) for details on what to include and how to link correctly with:

	setup_exponential.py
	setup_fixed.py

## Building

The following commands will build the libraries and link `c++/boost` to `python`:

	python setup_fixed.py build_ext --inplace
	python setup_exponential.py build_ext --inplace
