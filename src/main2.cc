#include <iostream>
#include <stdio.h>
#include <cmath>
#include <mpi.h>

#include "complex.h"
#include "input_image.cc"

#define PI 3.14159265358979f

void fourierTransform(Complex* array, size_t size) {
    if(size >= 2) {
        
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

        fourierTransform(array, size / 2);
        fourierTransform(array + size / 2, size / 2);

        float theta;
        for(int k = 0; k < size / 2; k++) {
            Complex even(array[k].real, array[k].imag);
            Complex odd(array[k + size / 2].real, array[k + size / 2].imag);
            theta = 2 * PI * k / size;
            Complex twiddle(cos(theta), -sin(theta));
            array[k] = even + twiddle * odd;
            array[k + size / 2] = even - twiddle * odd;
        }
    }
}


int main(int argc, char** argv) {
    
    InputImage image(argv[2]);
    
	int height = image.get_height();
	int width = image.get_width();
    int imgSize = height * width;
    

    Complex* img = image.get_image_data();
    Complex imgOutput[imgSize];

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int begin = (height / world_size) * world_rank;
    const int end = begin + (height / world_size);

    
    for (int i = begin; i < end; i++){
        fourierTransform(img + i * width, width);
    }

    for(int r = 0; r < height; ++r) {
        for(int c = 0; c < width; ++c) {
            std::cout << img[r * width + c] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}