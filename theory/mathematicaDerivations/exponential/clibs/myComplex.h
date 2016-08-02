#include <complex>

std::complex<double> Complex(double x, double y){
	const std::complex<double> i(0.0,1.0);
	return x + y*i;
}