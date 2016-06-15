Unit Tests
===============
This directory contains the unit tests for various important routines.

# Running Tests
Tests will **not** run in this directory as they require modules from other parallel directories.
Tests should therefore be run form the root repository directory.

## Run Tests
The following commands will run all tests from the root directory:

 - [`nosetest`](http://nose.readthedocs.io/en/latest/)
  * `nosetest -s` :: run with all output
  * `nosetest`    :: run with output on errors
 - `python tests` :: the directory can be directly called, running all tests
 
## Individual Tests
Individual tests may be run from the root repository as:

    python test/test_hmc.py