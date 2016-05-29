#Hybrid Monte Carlo & Kramer's Algorithm

This repository contains the code used for the thesis component of a Masters degree in Theoretical Physics at Edinburgh University.

Supervisors: Brian Pendleton, Tony Kennedy

## Hybrid Monte Carlo (HMC)
 - [Wikipedia Link](https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo)
 - [Seminal Paper (Duane, Kennedy, Pendleton)](http://www.sciencedirect.com/science/article/pii/037026938791197X)
 
HMC is used for sampling high dimensional probability distributions e.g. Lattice QCD where the space can be in excess of a million dimensions. 

The algorithm is highly effective by utilising Hamiltonian Dynamics after introducing a momentum field conjugate to the probability space that is refreshed after each sampler move. By utilising intrinsic gradient information provided by the geometry of the Hamiltonian, the sampler can transition through highly non-trivial spaces with exceptional efficiency when compared with the traditional Metropolis-Hastings approach.

## Kramer's Algorithm (KHMC)
 - [Kramer's Algorithm (Horowitz)](http://www.sciencedirect.com/science/article/pii/0370269391908125)
 
Kramer's Algorithm, also known as (L2MC), introduces an alternative approach whereby the conjugate momentum field is only partially refreshed after each sampler move.

## Generalised HMC (GHMC)
 - [Generalised Hybrid Monte Carlo (Kennedy, Pendleton)](http://www2.ph.ed.ac.uk/~adk/exact.pdf)

Both the above algorithms are specific parameterisations of the generalised Hybrid Monte Carlo algorithm.

## Conventions
Functions :: `CamelType()`
variables :: `lower_case`
Classes   :: `Upper_Case()`