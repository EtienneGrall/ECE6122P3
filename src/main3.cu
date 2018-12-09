#include <iostream>
#include <stdio.h>
#include <fstream>     
#include <string>
#include <cmath>

#include "complex.h"
#include "input_image.cc"

#define T_P_B 1024
#define PI 3.14159265358979f

__global__ void fourierTransform(Complex *img, Complex *imgTransformed, int width, int height) {

    int imgSize = width * height;
    
    if (threadIdx.x < imgSize){

        int quotient = imgSize / T_P_B;
        int modulo = imgSize % T_P_B;
        bool extra = modulo > threadIdx.x;
        int start = (quotient + (extra ? 1 : 0)) * threadIdx.x + (extra ? 0 : modulo);
        int end = start + quotient + (quotient + (extra ? 1 : 0));

        int k = blockIdx.x / width;
        int l = blockIdx.x % width;
        for (int i = start; i < end; i++){
            int m = i / width;
            int n = i % width;
            float theta = 2 * PI * (k * m + l * n) / width;
            Complex term(cos(theta), -sin(theta));
            imgTransformed[blockIdx.x] = imgTransformed[blockIdx.x] + img[i] * term;
        }
    }
    
}

int main(int argc, char** argv) {

	InputImage image(argv[1]);

	int h = image.get_height();
	int w = image.get_width();
    int imgSize = h * w;
    
    Complex* img = image.get_image_data();
    Complex imgTransformed[imgSize];

    Complex *d_img, *d_imgTransformed;
    const int memorySize = imgSize * sizeof(Complex);

    cudaMalloc((void**)&d_img, memorySize);
    cudaMalloc((void**)&d_imgTransformed, memorySize);
    
    cudaMemcpy(d_img, img, memorySize, cudaMemcpyHostToDevice);

    fourierTransform<<<imgSize, T_P_B>>>(d_img, d_imgTransformed, w, h);

    cudaMemcpy(imgTransformed, d_imgTransformed, memorySize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_img);
    cudaFree(d_imgTransformed);

    image.save_image_data(argv[2], imgTransformed, w, h);

    return 0;
}