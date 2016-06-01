Hybrid Monte Carlo (HMC)
===============
This directory contains the HMC code and test cases.

## Table of Contents
 - [To Do](#to-do)
 - [Unit Tests](#unit-tests)
     * [Hamiltonian Dynamics (Leap-Frog)](#hdlf)
 - [Code Acknowledgements](#ak)

<a name="to-do"/>
## To Do
 - ~~Hamiltonian Dynamics: Leap-Frog Integration~~
 - ~~Unit Test: Simple Harmonic Oscillator~~
 - HMC Sampling
 - Unit Test: Bivariate Gaussian
 - *Code on GPU (if time) & Unit Test*
 - *Unit Test (if time): v. high-dim Gaussian*

<a name="tests"/>
## Unit Tests

<a name="hdlf"/>
### Hamiltonian Dynamics (Leap-Frog)
<img src="./animations/ham_dynamics.gif" width="500" height="500" />
<img src="./plots/energy_drift.png" width="500" height="500" />

<a name="ak"/>
## Code Acknowledgements
 - `matlab_HMC.m` is taken from [The Clever Machine](https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/)
 - `theano_HMC.py` is taken from the DeepLearning.net [tutorial on HMC](http://deeplearning.net/tutorial/hmc.html)