#include <iostream>
#include <stdio.h>
#include <fstream>     
#include <string>
#include <cmath>

#include "complex.h"
#include "input_image.cc"

#define T_P_B 1024
#define PI 3.14159265358979f

__global__ void fourierTransformRow(Complex *img, Complex *imgTemp, int width, int height) {

    int rowIndex = blockIdx.x * T_P_B + threadIdx.x;
    if (rowIndex < height){
        float theta;
        int index;
        for (int n = 0; n < width; n++){
            index = rowIndex * width + n;
            for (int k = 0; k < width; k++){
                theta = 2 * PI * n * k / width;
                Complex twiddle(cos(theta), -sin(theta));
                imgTemp[index] = imgTemp[index] + img[rowIndex * width + k] * twiddle;
            }
        }
    } 
}

__global__ void fourierTransformColumn(Complex *imgTemp, Complex *imgTransformed, int width, int height) {

    int colIndex = blockIdx.x * T_P_B + threadIdx.x;
    if (colIndex < width){
        float theta;
        int index;
        for (int n = 0; n < height; n++){
            index = n * width + colIndex;            
            for (int k = 0; k < height; k++){
                theta = 2 * PI * n * k / height;
                Complex twiddle(cos(theta), -sin(theta));
                imgTransformed[index] = imgTransformed[index] + imgTemp[k * width + colIndex] * twiddle;
            }
        }
    } 
}

__global__ void inverseFourierTransformRow(Complex *imgTransformed, Complex *imgTemp, int width, int height) {

    int rowIndex = blockIdx.x * T_P_B + threadIdx.x;
    if (rowIndex < height){
        float theta;
        int index;
        for (int n = 0; n < width; n++){
            index = rowIndex * width + n;
            for (int k = 0; k < width; k++){
                theta = 2 * PI * n * k / width;
                Complex twiddle(cos(theta), sin(theta));
                imgTemp[index] = imgTemp[index] + imgTransformed[rowIndex * width + k] * twiddle;
            }
            imgTemp[index].real = imgTemp[index].real / width;
            imgTemp[index].imag = imgTemp[index].imag / width;
        }
    } 
}

__global__ void inverseFourierTransformColumn(Complex *imgTemp, Complex *img, int width, int height) {

    int colIndex = blockIdx.x * T_P_B + threadIdx.x;
    if (colIndex < width){
        float theta;
        int index;
        for (int n = 0; n < height; n++){
            index = n * width + colIndex;
            for (int k = 0; k < height; k++){
                theta = 2 * PI * n * k / width;
                Complex twiddle(cos(theta), sin(theta));
                img[index] = img[index] + imgTemp[k * width + colIndex] * twiddle;
            }
            img[index].real = img[index].real / height;
            img[index].imag = img[index].imag / height;
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

    Complex *d_img, *d_imgTemp, *d_imgTransformed;
    const int memorySize = imgSize * sizeof(Complex);

    cudaMalloc((void**)&d_img, memorySize);
    cudaMalloc((void**)&d_imgTemp, memorySize);
    cudaMalloc((void**)&d_imgTransformed, memorySize);
    
    

    if (strcmp(argv[1], "forward") == 0){
        cudaMemcpy(d_img, img, memorySize, cudaMemcpyHostToDevice);
        fourierTransformRow<<<h / T_P_B + 1, T_P_B>>>(d_img, d_imgTemp, w, h);
        fourierTransformColumn<<<w / T_P_B + 1, T_P_B>>>(d_imgTemp, d_imgTransformed, w, h);
        cudaMemcpy(imgOutput, d_imgTransformed, memorySize, cudaMemcpyDeviceToHost);
    }

    if (strcmp(argv[1], "reverse") == 0){
        cudaMemcpy(d_imgTransformed, img, memorySize, cudaMemcpyHostToDevice);
        inverseFourierTransformRow<<<h / T_P_B + 1, T_P_B>>>(d_imgTransformed, d_imgTemp, w, h);
        inverseFourierTransformColumn<<<w / T_P_B + 1, T_P_B>>>(d_imgTemp, d_img, w, h);
        cudaMemcpy(imgOutput, d_img, memorySize, cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_img);
    cudaFree(d_imgTemp);
    cudaFree(d_imgTransformed);

    image.save_image_data(argv[3], imgOutput, w, h);

    return 0;
}