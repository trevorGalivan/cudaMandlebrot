#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Complex double mumber type
class Cplx
{
public:
	double re;
	double im;
	__device__
	Cplx();

	__device__
	Cplx(double real);

	__device__
	Cplx(double real, double imag);

	__device__
	Cplx operator + (Cplx const& obj);

	__device__
	Cplx operator * (Cplx const& obj);

};


__device__
Cplx operator * (double lhs, const Cplx& rhs);

__device__
Cplx square(Cplx Z);

// returns square of magnitude of argument
__device__
double norm(Cplx Z);

__device__
double abs(Cplx rect);

// returns phase of argument, counter-clockwise from +'ve x axis
__device__
double phase(Cplx rect);

__device__
Cplx polar(Cplx rect); // Converts argument to polar form (magnitude, phase)

__device__
Cplx rect(Cplx polar); // Converts argument to rectangular form (real, imaginary)
