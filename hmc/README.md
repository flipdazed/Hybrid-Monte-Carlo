Hybrid Monte Carlo (HMC)
===============
This directory contains the HMC code and test cases.

## Table of Contents
 - [To Do](#td)
 - [Interesting Tests](#tests)
     * [Hamiltonian Dynamics (Leap-Frog)](#tests-hdlf)
	 * [Leap-Frog Parameters](#tests-lf)
	 * [HMC sampling Bivariate Gaussian](#tests-bg)
 - [Code Acknowledgements](#ak)

<a name="td"/>
## To Do
 - ~~Hamiltonian Dynamics: Leap-Frog Integration~~
 - ~~Practical Test: Simple Harmonic Oscillator~~
 - ~~Unit Test: Energy Conservation~~
 - ~~Unit Test: Reversibility: Satisfy Detailed Balance~~
 - ~~HMC Sampler~~
 - ~~Unit Test: Simple Harmonic Oscillator (1D Gaussian)~~
 - ~~Unit Test: Bivariate Gaussian~~
 - ~~Quantum Harmonic Oscillator Potential for Lattice QFT~~
 - Generalise to Euclidean time for Lattice QFT
 	* ~~Gradient / Laplacian functions on the lattice~~
 	* ~~Unit Test: Gradient / Laplacian vs. expected values~~
 	* ~~Leap Frog algorithm on the lattice (position and momentum updates)~~
 	* Unit Test: Leap Frog moves on the lattice
 	* HMC on the Lattice - minimal changes: all in Leap Frog
 	* Unit Test: HMC on the Lattice
 - Practical Test: Quantum Harmonic Oscillator
 	* ground state
 	* two-point correlation functions
 - Practical Test: Quantum Anharmonic Oscillator phi^4
 	* ground state
 	* two-point correlation functions
 - Extend to Kramers condition

 - *Code on GPU (if time) & Unit Test*
 - *Unit Test (if time): v. high-dim Gaussian*

<a name="tests"/>
## Interesting Tests

<a name="tests-hdlf"/>
### Hamiltonian Dynamics (Leap-Frog)
A basic demonstration of symplectic Leap-Frog integrator. The key points to note are the
conservation of energy shown by the circular phase space path and near stationary hamiltonian. The small oscillations are due to *energy drift* which can also be seen in the subtle deviation from a true circle in phase space. Importantly the circular phase path is a closed loop showing reversibility in artificial time.

Animated Summary | Energy Drift
:---:|:---:
<img src="./animations/ham_dynamics.gif" width="500" height="500" />  |  <img src="./plots/energy_drift.png" width="500" height="500" />

<a name="tests-lf"/>
### Energy Drift for varying Leap-Frog Parameters
A contour plot showing the small energy loss encountered for different Leap-Frog parameters. Tested using the Simple Harmonic Oscillator. Acceptable tolerance level is
show as a dashed horizontal line.
<p align="center">
	<img src="./plots/energy_conservation.png" width="700" height="700" />
</p>


<a name="tests-bg"/>
### Sampling a Bivariate Gaussian
A demonstration of the first 50 HMC moves when sampling a bivariate gaussian distribution. Normally these moves would be part of the *burn in* phase. An interesting thing to note is that this is equivalent to a simple harmonic oscillator based in a point-like lattice (i.e. no lattice!) of two dimensions. The covariance matrix can be diagonalised and linearly decomposed into two separate oscillators and then they can be considered as two separated point oscillators.

<p align="center">
	<img src="./plots/HMC_gauss_2d.png" width="700" height="700" />
</p>

<a name="ak"/>
## Code Acknowledgements
 - `matlab_HMC.m` is taken from [The Clever Machine](https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/)
 - `theano_HMC.py` is taken from the DeepLearning.net [tutorial on HMC](http://deeplearning.net/tutorial/hmc.html)
 - `mh_qho.py` is taken from Lepage, G. Peter. 1998. [Lattice QCD for novices](http://arxiv.org/abs/hep-lat/0506036)