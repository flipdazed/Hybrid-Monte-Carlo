import numpy as np
import os
import json, codecs, base64
import dill as pickle

DATA_LOC = 'results/data/'
wparams = {} # {'encoding':'utf-8'} # writing params
rparams = {} # the above messes it up
flocs = {'numpy_objs':'.json', 'other_objs':'.pkl'} # locations and extensions
# there is a hardcoded value in function load() refrencing str :: 'numpy_objs
# because tbh I'm tired and hungry and cba to waste time on loading / saving data

class NumpyEncoder(json.JSONEncoder):
    """Encodes a numpy array in JSON format while preserving the data format
    
    http://stackoverflow.com/a/24375113/4013571
    """
    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict 
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)
#
def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype
    
    Required Inputs
        dct :: dict :: json encoded ndarray
    
    Returns
        (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct
#
def writefile(fname, obj_id):
    """Gets the correct file open
    
    Required Inputs
        path :: string :: file path
        id   :: string :: must be in ['np', 'else']
    """
    
    # check input is sensible
    if obj_id not in flocs.keys(): raise ValueError(
        'obj_id:{} not in {}'.format(obj_id, flocs.keys()))
    
    fname = os.path.splitext(fname)[0]
    path = os.path.join(DATA_LOC, obj_id, fname + flocs[obj_id])
    hook = codecs.open(path, 'w', **wparams)
    return path, hook
#
def load(path):
    """opens the file with the correct method
    
    Required Inputs
        path :: string :: file path
    """
    
    # check the file referenced is sensible
    obj_id = [k for k in flocs.keys() if k in path]
    if obj_id is None or len(obj_id) > 1: raise ValueError(
        '{} not found in the path: \n {}'.format(flocs.keys(), path))
    obj_id = obj_id.pop(0)
    
    with codecs.open(path, 'r', **rparams) as f:
        if obj_id == 'numpy_objs':
            obj = json.load(f, object_hook=json_numpy_obj_hook)
        else:
            obj = pickle.load(file=f)
    return obj
#
def store(obj, filename):
    """Stores a python object
    
    Required Inputs
        obj :: anything! :: stores this object
        path :: str :: the save name in ./data/
    """
    # It is a numpy array
    if type(obj) == np.ndarray:
        path,f = writefile(filename, obj_id='numpy_objs')
        json.dump(obj, fp=f, cls=NumpyEncoder,
            separators=(',', ':'), sort_keys=True, indent=4)
        print '> saved with JSON to {}'.format(path)
    else:
        path, f = writefile(filename, obj_id='other_objs')
        pickle.dump(obj, file=f)
        print '> saved with dill (pickled) to {}'.format(path)
    return path
#
if __name__ == '__main__':
    expected = np.arange(100, dtype=np.float)
    
    dumped = json.dumps(expected, cls=NumpyEncoder)
    result = json.loads(dumped, object_hook=json_numpy_obj_hook)
    
    assert result.dtype == expected.dtype, "Wrong Type"
    assert result.shape == expected.shape, "Wrong Shape"
    assert np.allclose(expected, result), "Wrong Values"
    
    expected = np.arange(100, dtype=np.float)
    path   = store(expected, 'test')
    result = load(path)
    
    assert result.dtype == expected.dtype, "Wrong Type"
    assert result.shape == expected.shape, "Wrong Shape"
    assert np.allclose(expected, result), "Wrong Values"
    
    expected = range(100)
    
    path   = store(expected, 'test')
    result = load(path)
    
    assert type(result) == type(expected), "Wrong Type"
    assert len(result)  == len(expected), "Wrong Shape"
    assert expected     == result, "Wrong Values"