//
// Created by brian on 11/20/18.
//

#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <cmath>
#include <cuda.h>

const float PI = 3.14159265358979f;

class Complex {
public:
    Complex(): real(0.0f), imag(0.0f) {}
    Complex(float r): real(r), imag(0.0f) {}
    CUDA_HOSTDEV Complex(float r, float i): real(r), imag(i) {}

    CUDA_HOSTDEV Complex operator+(const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }

    Complex operator-(const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }

    CUDA_HOSTDEV Complex operator*(const Complex& b) const {
        return Complex(real * b.real  - imag * b.imag, real * b.imag + imag * b.real);
    }


    float mag() const {
        return sqrt(pow(real, 2) + pow(imag, 2));
    }

    float angle() const {
        if (real > 0){
            return atan(real / imag);
        } else if (real < 0) {
            if (imag >= 0){
                return atan(real / imag) + PI;
            } else {
                return atan(real / imag) - PI;
            }
        } else {
            if (imag > 0){
                return PI / 2;
            } else if (imag < 0){
                return - PI / 2;
            }
        } 
        return 0;
    }

    Complex conj() const {
        return Complex(real, -imag);
    }


    float real;
    float imag;
};

std::ostream& operator<<(std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}

