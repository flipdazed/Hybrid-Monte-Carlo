#include <iostream>
#include <iomanip>
#include "ghmc1AccNormalised.hpp"
#include "hmc1AccNormalised.hpp"

int main()
{
	double pi    = 3.141592653589793238463;
	double t     = 10;
	double theta = pi/2.0;
	double dtau  = 0.1;
	double n_steps = 20;
	double m    = 1.0;
	double tau;
	double phi;
	std::complex<double> hmc;
	std::complex<double> ghmc;
	double r;
	
	tau  = n_steps*dtau;
	phi  = tau*m;
	r = 1.0/tau;
	ghmc = ghmc1AccNormalised(phi, r, 2.0, pi/2);
	hmc  =  hmc1AccNormalised(phi, r, 2.0);
	std::cout	<< "Hello World" 	<< std::endl;
	std::cout	<< std::setprecision(5)
				<< ghmc	<< std::endl;
	std::cout	<< hmc	<< std::endl;
	std::cout	<< "Goodby World" 	<< std::endl;

	return 0;
}