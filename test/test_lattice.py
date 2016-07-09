import numpy as np

import utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.lattice import Periodic_Lattice, laplacian, gradSquared
from scipy.ndimage.filters import laplace
class Test(object):
    def __init__(self):
        self.id = 'lattice'
        self.a1 = np.array([[ 11.,  12.,  13.,  14.],
                            [ 21.,  22.,  23.,  24.],
                            [ 31.,  32.,  33.,  34.],
                            [ 41.,  42.,  43.,  44.]])
        
        self.l = Periodic_Lattice(self.a1, lattice_spacing=1.)
        
    def wrap(self, print_out = True):
        """tests the wrapping function against expected values"""
        passed = True
        a = self.a1 # shortcut
        passed *= (self.l == self.a1).all() # check both are equal
        test = [[(1,1), 22.], [(3,3), 44.], [(4,4), 11.], # [index, expected value]
            [(3,4), 41.], [(4,3), 14.], [(10,10), 33.]]
        
        for idx, act in test: # iterate test values
            passed *= (self.l[idx] == act)
        
        if print_out:
            utils.display('Periodic Boundary', outcome=passed,
                details = {'array storage checked':[],
                    'period indexing vs. known values':[]})
        
        return passed
    
    def laplacian(self, print_out = True):
        """tests the laplacian function against expected values"""
        passed = True
        a = self.a1 # shortcut
        passed *= (self.l == self.a1).all() # check both are equal
        test = [[(1,1), np.asarray([  0.,  0.]).sum()], # 11
                [(3,3), np.asarray([-40., -4.]).sum()], # 44
                [(4,4), np.asarray([ 40.,  4.]).sum()], # 11
                [(3,4), np.asarray([-40.,  4.]).sum()], # 41
                [(4,3), np.asarray([ 40., -4.]).sum()], # 14
                [(2,3), np.asarray([  0., -4.]).sum()]] # 34
        
        store = []
        for pos, act in test: # iterate test values
            res = laplacian(self.l, position=pos, a_power=0)
            passed *= (res == act).all()
            if print_out: store.append([pos, res, act])
        
        if print_out:
            utils.display('Laplacian', outcome=passed,
                details = {'checked vs. known values (Mathematica)':[
                    'pos: {}, res: {}, act: {}'.format(*vals) for vals in store
                ]})
        
        return passed
    
    def sciPyLaplacian(self, print_out = True):
        """tests the laplacian function against expected values"""
        passed = True
        a = self.a1 # shortcut
        test = [[(1,1), np.asarray([  0.,  0.]).sum()], # 11
                [(3,3), np.asarray([-40., -4.]).sum()], # 44
                [(4,4), np.asarray([ 40.,  4.]).sum()], # 11
                [(3,4), np.asarray([-40.,  4.]).sum()], # 41
                [(4,3), np.asarray([ 40., -4.]).sum()], # 14
                [(2,3), np.asarray([  0., -4.]).sum()]] # 34
        
        store = []
        for pos, act in test: # iterate test values
            res = laplace(a, mode='wrap').view(Periodic_Lattice)[pos]
            passed *= (res == act).all()
            if print_out: store.append([pos, res, act])
        
        if print_out:
            utils.display('SciPy Laplacian', outcome=passed,
                details = {'checked vs. known values (Mathematica)':[
                    'pos: {}, res: {}, act: {}'.format(*vals) for vals in store
                ]})
        
        return passed
    
    def gradSquared(self, print_out = True):
        """tests the gradient squared function against expected values"""
        passed = True
        a = self.a1 # shortcut
        passed *= (self.l == self.a1).all() # check both are equal
        
        test = [[(1,1), np.square(np.asarray([ 10.,   1.])).sum()],  # 11
                [(3,3), np.square(np.asarray([-30., - 3.])).sum()],  # 44
                [(4,4), np.square(np.asarray([ 10.,   1.])).sum()],  # 11
                [(3,4), np.square(np.asarray([-30.,   1.])).sum()],  # 41
                [(4,3), np.square(np.asarray([ 10., - 3.])).sum()],  # 14
                [(2,3), np.square(np.asarray([ -3.,  10.])).sum()]]  # 34
        
        store = []
        for pos, act in test: # iterate test values
            res = gradSquared(self.l, position = pos, a_power = 0)
            passed *= (res == act).all()
            if print_out: store.append([pos, res, act])
        
        if print_out:
            utils.display('Gradient Squared : (forward diff)', outcome=passed,
                details = {'checked vs. known values (Mathematica)':[
                    'pos: {}, res: {}, act: {}'.format(*vals) for vals in store
                ]})
        
        return passed
#
if __name__ == '__main__':
    test = Test()
    utils.newTest(test.id)
    test.sciPyLaplacian()
    test.wrap()
    test.laplacian()
    test.gradSquared()