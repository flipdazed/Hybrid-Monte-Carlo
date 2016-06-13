# -*- coding: utf-8 -*-


# The program below is extracted from the following reference:

# Lepage, G. Peter. 1998. Lattice QCD for novices. In <em>Strong Interactions
# at Low and Intermediate Energies: Proceedings of the 13th Annual HUGS
# at CEBAF, Jefferson Laboratory</em> (edited by  J. L. Goity), pp. 49â€“90.
# River Edge, N.J.: World Scientific.
# Preprint available at arxiv.org/abs/hep-lat/0506036

# To understand what the code does and how to interpret the results, please
# see the Lepage article.

# The only changes made in the published program are minor modifications
# needed to make it compatible with more-recent versions of Python. In
# NumPy package.
# particular, the older Numeric package has been replaced with the modern


import numpy 
from random import uniform 
from math import * 


def update(x): 
    for j in range(0,N): 
        old_x = x[j]                        # save original value 
        old_Sj = S(j,x) 
        x[j] = x[j] + uniform(-eps,eps)     # update x[j] 
        dS = S(j,x) - old_Sj                # change in action 
        if dS > 0 and exp(-dS) < uniform(0,1): 
            x[j] = old_x                    # restore old value
 
def S(j,x):         # harm. osc. S 
    jp = (j+1) % N  # next site
    jm = (j-1) % N  # previous site
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

def compute_G(x,n): 
    g = 0 
    for j in range(0,N): 
        g = g + x[j]*x[(j+n) % N] 
    return g/N

def MCaverage(x,G): 
    for j in range(0,N):            # initialize x 
        x[j] = 0 
    for j in range(0,5*N_cor):      # thermalize x 
        update(x) 
    for alpha in range(0,N_cf):     # loop on random paths 
        for j in range(0,N_cor): 
            update(x)
        for n in range(0,N): 
            G[alpha][n] = compute_G(x,n) 
    for n in range(0,N):            # compute MC averages 
        avg_G = 0 
        for alpha in range(0,N_cf): 
            avg_G = avg_G + G[alpha][n] 
        avg_G = avg_G/N_cf 
        print "G(%d) = %g" % (n,avg_G) 

def bootstrap(G): 
    N_cf = len(G) 
    G_bootstrap = []                    # new ensemble 
    for i in range(0,N_cf): 
        alpha = int(uniform(0,N_cf))    # choose random config 
        G_bootstrap.append(G[alpha])    # keep G[alpha] 
    return G_bootstrap 

def bin(G,binsize): 
    G_binned = []                       # binned ensemble 
    for i in range(0,len(G),binsize):   # loop on bins 
        G_avg = 0 
        for j in range(0,binsize):      # loop on bin elements 
            G_avg = G_avg + G[i+j] 
        G_binned.append(G_avg/binsize)  # keep bin avg 
    return G_binned 


# set parameters: 

N = 20 
N_cor = 20 
N_cf = 1000
a = 0.5 
eps = 1.4 

# create arrays:
 
x = numpy.zeros((N,), dtype=float) 
G = numpy.zeros((N_cf,N), dtype=float) 

# do the simulation: 

MCaverage(x,G) 

# If the file is called qcd.py, it is run with the command:
#     'python qcd.py' 
# 
# To test the binning and bootstrap codes add the following
# to the the file: 

def avg(G):         # MC avg of G 
    return numpy.sum(G,axis=0)/len(G) 

def sdev(G):        # std dev of G 
    g = numpy.asarray(G) 
    return numpy.absolute(avg(g**2)-avg(g)**2)**0.5

print 'avg G\n',avg(G) 
print 'avg G (binned)\n',avg(bin(G,4)) 
print 'avg G (bootstrap)\n',avg(bootstrap(G)) 

# The average of the binned copy of G should be the same as the 
# average of G itself; the average of the bootstrap copy should 
# be different by an amount of order the Monte Carlo error. 
# Compute averages for several bootstrap copies to get a good 
# feel for the errors. Finally one wants to extract energies. 
# This is done by adding code to compute âˆ†E(t): 

def deltaE(G):      # Delta E(t)
    avgG = avg(G)
    adE = numpy.log(numpy.absolute(avgG[:-1]/avgG[1:]))
    return adE/a 

print 'Delta E\n',deltaE(G) 
# print 'Delta E (bootstrap)\n',deltaE(bootstrap(G)) 

# Again repeating the evaluation for 50 or 100 bootstrap 
# copies of G gives an estimate of the statistical errors 
# in the energies. Additional code can be added to evaluate 
# standard deviations from these copies: 

def bootstrap_deltaE(G,nbstrap=100):    # Delta E + errors 
    avgE = deltaE(G)                # avg deltaE 
    bsE = [] 
    for i in range(nbstrap):    # bs copies of deltaE 
        g = bootstrap(G) 
        bsE.append(deltaE(g)) 
    bsE = numpy.array(bsE) 
    sdevE = sdev(bsE)               # spread of deltaE's 
    print "\n%2s %10s %10s" % ("t","Delta E(t)","error") 
    print 2 *"-" 
    for i in range(len(avgE)/2): 
        print "%2d %10g %10g" % (i,avgE[i],sdevE[i]) 

bootstrap_deltaE(G)