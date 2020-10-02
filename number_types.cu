#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "number_types.cuh"

__device__
Cplx::Cplx() {
	re = 0.;
	im = 0.;
}

__device__
Cplx::Cplx(double real) {
	re = real;
	im = 0.;
}

__device__
Cplx::Cplx(double real, double imag) {
	re = real;
	im = imag;
}

__device__
Cplx Cplx::operator + (Cplx const& obj) {
	Cplx res;
	res.re = re + obj.re;
	res.im = im + obj.im;
	return res;
}
__device__
Cplx Cplx::operator * (Cplx const& obj) {
	Cplx res;
	res.re = re * obj.re - im * obj.im;
	res.im = re * res.im + im * res.re;
	return res;
}

__device__
Cplx operator * (double lhs, const Cplx& rhs) {
	Cplx res;
	res.re = rhs.re * lhs;
	res.im = rhs.im * lhs;
	return res;
}

__device__
Cplx square(Cplx Z) {
	return Cplx(Z.re * Z.re - Z.im * Z.im, 2. * Z.re * Z.im);
}

// returns square of magnitude of argument
__device__
double norm(Cplx Z) {
	return Z.re * Z.re + Z.im * Z.im;
}

__device__
double abs(Cplx rect) {
	return sqrt(rect.re * rect.re + rect.im * rect.im);
}

// returns phase of argument, counter-clockwise from +'ve x axis
__device__
double phase(Cplx rect) {
	return atan2(rect.im, rect.re);
}

__device__
Cplx polar(Cplx rect) { // Converts argument to polar form (magnitude, phase)
	return Cplx(abs(rect), phase(rect));
}

__device__
Cplx rect(Cplx polar) { // Converts argument to rectangular form (real, imaginary)
	return Cplx(polar.re * cos(polar.im), polar.re * sin(polar.im));
}

/*
class Cplx
{
public:
	double re;
	double im;
	__device__
	Cplx() {
		re = 0.;
		im = 0.;
	}
	__device__
	Cplx(double real) {
		re = real;
		im = 0.;
	}

	__device__
	Cplx(double real, double imag) {
		re = real;
		im = imag;
	}
	__device__
	Cplx operator + (Cplx const& obj) {
		Cplx res;
		res.re = re + obj.re;
		res.im = im + obj.im;
		return res;
	}
	__device__
	Cplx operator * (Cplx const& obj) {
		Cplx res;
		res.re = re * obj.re - im * obj.im;
		res.im = re * res.im + im * res.re;
		return res;
	}

};


__device__
Cplx operator * (double lhs, const Cplx& rhs) {
	Cplx res;
	res.re = rhs.re * lhs;
	res.im = rhs.im * lhs;
	return res;
}

__device__
Cplx square(Cplx Z) {
	return Cplx(Z.re * Z.re - Z.im * Z.im, 2. * Z.re * Z.im);
}

// returns square of magnitude of argument
__device__
double norm(Cplx Z) {
	return Z.re * Z.re + Z.im * Z.im;
}

__device__
double abs(Cplx rect) {
	return sqrt(rect.re * rect.re + rect.im * rect.im);
}

// returns phase of argument, counter-clockwise from +'ve x axis
__device__
double phase(Cplx rect) {
	return atan2(rect.im, rect.re);
}

__device__
Cplx polar(Cplx rect) { // Converts argument to polar form (magnitude, phase)
	return Cplx(abs(rect), phase(rect));
}

__device__
Cplx rect(Cplx polar) { // Converts argument to rectangular form (real, imaginary)
	return Cplx(polar.re * cos(polar.im), polar.re * sin(polar.im));
}


*/