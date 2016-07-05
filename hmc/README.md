Hybrid Monte Carlo (HMC)
===============
This directory contains the HMC code and test cases.

# Table of Contents
 - [To Do](#td)
     * [1D Lattice Quantum Theory](#td-1d)
	 * [nD Lattice Quantum Field Theory](#td-nd)
 - [Interesting Tests](#tests)
     * [Hamiltonian Dynamics (Leap-Frog)](#tests-hdlf)
	 * [Leap-Frog Parameters](#tests-lf)
	 * [HMC sampling Bivariate Gaussian](#tests-bg)
	 * [Sampling the QHO](#tests-qho)
	 * [Acceptance Rates and Hamiltonian Change](#tests-acc)
 - [Code Acknowledgements](#ak)

<a name="td"/>
# To Do

<a name="td-1d"/>
## 1D Continuum
### Implementation
 - ~~Hamiltonian Dynamics: Leap-Frog Integration~~
 - ~~Practical Test: Simple Harmonic Oscillator~~
 - ~~Unit Test: Energy Conservation~~
 - ~~Unit Test: Reversibility: Satisfy Detailed Balance~~
 - ~~HMC Sampler~~

### Results
 - ~~Unit Test: Simple Harmonic Oscillator (1D Gaussian)~~
 - ~~Unit Test: Bivariate Gaussian~~
 - ~~Plot: Sampled MVG Potential for 2D (easy gen. to nD)~~
 - ~~Plot: Sampled SHO Potential for 1D~~

<a name="td-nd"/>
## nD Lattice Quantum Field Theory
### Implementation
 - ~~Gradient / Laplacian functions on the lattice~~
 - ~~Unit Test: Gradient / Laplacian vs. expected values~~
 - ~~Potentials on the Lattice~~
 - ~~Leap Frog algorithm on the lattice (position and momentum updates)~~
 - ~~Unit Test: Leap Frog moves on the lattice~~
 - ~~HMC on the Lattice - minimal changes: all in Leap Frog~~
 - ~~Unit Test: HMC on the Lattice~~

### Results
 - ~~Plot: Quantum Path for QHO~~
 - ~~Plot: Sampled Quantum Harmonic Oscillator potential~~
 - Plot: Quantum Path for AHO
 - Plot: Quantum Anharmonic Oscillator potential (phi^4)

<a name="tests"/>
## Interesting Tests

<a name="tests-hdlf"/>
### Hamiltonian Dynamics (Leap-Frog)
A basic demonstration of symplectic Leap-Frog integrator. The key points to note are the
conservation of energy shown by the circular phase space path and near stationary hamiltonian. The small oscillations are due to *energy drift* which can also be seen in the subtle deviation from a true circle in phase space. Importantly the circular phase path is a closed loop showing reversibility in artificial time.

Animated Summary | Energy Drift
:---:|:---:
<img src="../results/animations/ham_dynamics.gif" width="500" height="500" />  |  <img src="../results/figures/dynamics_constEn_1d_H_sho.png" width="500" height="500" />

<a name="tests-lf"/>
### Energy Drift for varying Leap-Frog Parameters
A contour plot showing the small energy loss encountered for different Leap-Frog parameters. Tested using the 1D free-field Klein Gordon potential.
<p align="center">
	<img src="../results/figures/dynamics_constEn_2d_kg.png" width="700" height="700" />
</p>


<a name="tests-bg"/>
### Sampling a Bivariate Gaussian
A demonstration of the first 50 HMC moves when sampling a bivariate gaussian distribution. The burn-in phase is shown in grey. An interesting thing to note is that this is equivalent to a simple harmonic oscillator based in a point-like lattice (i.e. no lattice!) of two dimensions. The covariance matrix can be diagonalised and linearly decomposed into two separate oscillators and then they can be considered as two separated point oscillators.

<p align="center">
	<img src="../results/figures/hmc_samplePlot_mvg.png" width="700" height="700" />
</p>


<a name="tests-qho"/>
### Sampling the Quantum Harmonic Oscillator
The 1D Quantum Harmonic Oscillator is sampled using the MCMC path integral method. This is shown to agree closely with theory.
<p align="center">
	<img src="../results/figures/hmc_sample_1d_qho.png" width="700" height="700" />
</p>

<a name="tests-acc">
### Two Point Correlation Function: QHO
Here the results of Creutz and Freedman are reproduced for the two-point correlation function for the Quantum harmonic Oscillator. A large number of HMC samples are required to obtain a good fit. This is likely related to the acceptance rate issue below.
<p align="center">
	<img src="../results/figures/corr1d_x2_qho.png" width="700" height="700" />
</p>

<a name="tests-acc">
### Acceptance Rates and change in Hamiltonian (QHO)
Here, in the top figure, is shown the acceptance probability for the Metropolis-Hastings step at the end of each Leap-Frog trajectory. The middle figure shows the old and new Hamiltonian at the end of each trajectory and the final, bottom, figure shows the exponentiated negative change in the Hamiltonian between the start and the end of each trajectory. Pink bands show where rejections occurred at run-time.
The optimal acceptance probability is around `60%` and `<exp{-𝛿H}>` should sit around `1.`. Currently there is an, as of yet, undiagnosed issue in the code made clear by this test.
<p align="center">
	<img src="../results/figures/metropolis_accRej_qho.png" width="700" height="700" />
</p>

<a name="ak"/>
## Code Acknowledgements
See `../ref_code` for code snippets I found useful: Always helpful to see existing implementations to guide new implementations.