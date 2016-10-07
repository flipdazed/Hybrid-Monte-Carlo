Hybrid Monte Carlo & Kramers Algorithm
===============

**NB: re-writing all the docstrings to fit `PEP` standards**
*doc strings are currently being revamped to meet `PEP` standards and allow a `numpy`-style website documenting the code base*

This repository contains the code used for the thesis component of a Masters degree in Theoretical Physics at Edinburgh University.

Supervisors: Brian Pendleton, Tony Kennedy

$\LaTeX$ is used widely in this repository. A good extension for Google Chrome is [`TeX All the Things`](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn)

## Table of Contents
 - [Hybrid Monte Carlo (HMC)](#hmc)
 - [Kramers Algorithm (KHMC)](#khmc)
 - [Generalised HMC (GHMC)](#ghmc)
 - [Conventions](#conv)

<a name="hmc"/>
## Hybrid Monte Carlo (HMC)
 - [Wikipedia Link](https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo)
 - [Seminal Paper (Duane, Kennedy, Pendleton)](http://www.sciencedirect.com/science/article/pii/037026938791197X)
 
HMC is used for sampling high dimensional probability distributions e.g. Lattice QCD where the space can be in excess of a million dimensions. 

The algorithm is highly effective by utilising Hamiltonian Dynamics after introducing a momentum field conjugate to the probability space that is refreshed after each sampler move. By utilising intrinsic gradient information provided by the geometry of the Hamiltonian, the sampler can transition through highly non-trivial spaces with exceptional efficiency when compared with the traditional Metropolis-Hastings approach.

<a name="khmc"/>
## Kramers Algorithm (KHMC)
 - [Kramers Algorithm (Horowitz)](http://www.sciencedirect.com/science/article/pii/0370269391908125)
 
Kramers Algorithm, also known as (L2MC), introduces an alternative approach whereby the conjugate momentum field is only partially refreshed after each sampler move.

<a name="ghmc"/>
## Generalised HMC (GHMC)
 - [Generalised Hybrid Monte Carlo (Kennedy, Pendleton)](http://www2.ph.ed.ac.uk/~adk/exact.pdf)

Both the above algorithms are specific parameterisations of the generalised Hybrid Monte Carlo algorithm.

<a name="conv"/>
## Conventions
- Functions :: `CamelType()`
- Variables :: `lower_case`
- Classes   :: `Upper_Case()`
