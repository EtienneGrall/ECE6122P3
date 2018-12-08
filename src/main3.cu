#include <iostream>
#include <stdio.h>
#include <fstream>     
#include <string>

#include "complex.cc"
#include "input_image.cc"

#define T_P_B 1024

__global__ void iterate() {
	printf("Hello from block %d thread %d", blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv) {

	InputImage image(argv[1]);

	int h = image.get_height();
	int w = image.get_width();
    int imgSize = h * w;
    
    Complex* img = image.get_image_data();
    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            std::cout << img[r * w + c] << " ";
        }
        std::cout << std::endl;
    }


    iterate<<<imgSize, T_P_B>>>();
    return 0;
}