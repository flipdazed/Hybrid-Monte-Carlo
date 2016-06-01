import numpy as np

class Simple_Harmonic_Oscillator(object):
    """Simple Harmonic Oscillator
    
    The potential is given by: F(x) = k*x
    
    Optional Inputs
        k :: float :: spring constant
    """
    def __init__(self, k=1.):
        self.k = k
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda x: self.gradPotentialEnergy(x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    def kineticEnergy(self, p):
        return p**2/2.
    def potentialEnergy(self, x):
        return self.k*x**2/2.
    def gradKineticEnergy(self, p):
        return p
    def gradPotentialEnergy(self, x):
        return self.k*x