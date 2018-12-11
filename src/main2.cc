#include <iostream>
#include <stdio.h>
#include <cmath>

#include "complex.h"
#include "input_image.cc"

#define WORLD_SIZE 8
#define PI 3.14159265358979f

void splitEvenOdd(Complex* array, int size) {

}

void fourierTransform(Complex* array, int size) {
    if(N >= 2) {
        
    	Complex temp[size / 2];

    	for(int i = 0; i < size / 2; i++){
    	    temp[i] = array[i * 2 + 1];
    	}
    	for(int i = 0; i < size / 2; i++){
    	    array[i] = array[i * 2];
    	}
    	for(int i = 0; i < size / 2; i++){
    	    array[i + size / 2] = temp[i];
    	}

        fourierTransform(X, N / 2);
        fourierTransform(X + N / 2, N / 2);

        for(int k = 0; k < N / 2; k++) {
            complex<double> e = X[k    ];   // even
            complex<double> o = X[k+N/2];   // odd
            complex<double> w = exp( complex<double>(0,-2.*M_PI*k/N) );
            X[k    ] = e + w * o;
            X[k+N/2] = e - w * o;
        }
    }
}


int main(int argc, char** argv) {
    InputImage image(argv[2]);

	int h = image.get_height();
	int w = image.get_width();
    int imgSize = h * w;
    

    Complex* img = image.get_image_data();
    Complex imgOutput[imgSize];
    return 0;
}