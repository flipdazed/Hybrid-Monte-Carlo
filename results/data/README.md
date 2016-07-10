Dataset Library
===============

Contains data that was generated to produce plots. 
Useful to edit plots without having to run long expensive simulations.

Optimises JSON where possible for `numpy` arrays. Other types are 
pickled by `dill`. `dill` has more flexibility than `cpickle` and allows
the storage of instantiated functions and `lambda` expressions. It is
slower than `cPickle` but the items to be stored are not large.

##`store.py`##
Example usage is given below with more inside `store.py`,

    expected = np.arange(100, dtype=np.float)
    path   = store(expected, 'test')
    result = load(path)
    
    assert result.dtype == expected.dtype, "Wrong Type"
    assert result.shape == expected.shape, "Wrong Shape"
    assert np.allclose(expected, result), "Wrong Values"

##data##
- `numpy_objs` contains saved `numpy` data in JSON format
- `other_objs` contains other data formats pickled with `dill`