#Hybrid Monte Carlo: Kramer's Algorithm

This repository contains the code used to generate results for the thesis component of a Masters degree in Theoretical Physics at Edinburgh University.

**Hybrid Monte Carlo (HMC)** is used for sampling high dimensional probability distributions e.g. Lattice QCD where the space can be in excess of a million dimensions. 

The algorithm is highly effective by utilising Hamiltonian Dynamics after introducing a momentum field conjugate to the probability space that is refreshed after each sampler move. By utilising intrinsic gradient information provided by the geometry of the Hamiltonian, the sampler can transition through highly non-trivial spaces with exceptional efficiency when compared with the traditional Metropolis-Hastings approach.

**Kramer's Algorithm** introduces an alternative approach whereby the conjugate momentum field is only partially refreshed after each sampler move.

**generalised HMC** both algorithms are specific parameterisations of the genralised Hybrid Monte Carlo algorithm

## Hybrid Monte Carlo
 - [Wikipedia Link](https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo)
 - [Seminal Paper](http://www.sciencedirect.com/science/article/pii/037026938791197X)
 
## Kramer's Algorithm and generalised HMC
 - [Kramer's Algorithm (L2MC), A. Horowitz](http://www.sciencedirect.com/science/article/pii/0370269391908125)
 - [Generalised Hybrid Monte Carlo](http://www2.ph.ed.ac.uk/~adk/exact.pdf)
