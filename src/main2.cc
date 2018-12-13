#include <iostream>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include <stddef.h>

#include "complex.h"
#include "input_image.cc"

using namespace std; 
#define PI 3.14159265358979f

void split(Complex* array, const size_t size)
{
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
}

void fourierTransform(Complex* array, const size_t size, const bool inverse, const bool normalize) 
{
    if(size >= 2) {
        
        split(array, size);

        fourierTransform(array, size / 2, inverse, false);
        fourierTransform(array + size / 2, size / 2, inverse, false);

        float theta;
        for(int k = 0; k < size / 2; k++) {
            Complex even(array[k].real, array[k].imag);
            Complex odd(array[k + size / 2].real, array[k + size / 2].imag);
            theta = 2 * PI * k / size;
            Complex twiddle(cos(theta), (inverse ? sin(theta) : -sin(theta)));
            array[k] = even + twiddle * odd;
            array[k + size / 2] = even - twiddle * odd;
            if (normalize)
            {
                const Complex factor(1.0/size);
                array[k] = array[k]*factor;
                array[k + size / 2] = array[k + size / 2]*factor;
            }
        }
    }
}

void transpose(Complex* array, const int width, const int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = i+1; j < height; j++)
        {
            Complex temp = array[i*width + j];
            array[i*width + j] = array[j*width + i];
            array[j*width + i] = temp;
        }
    }
}

void gatherFourierTransform(Complex* img, const bool inverse, const int begin, const int end, const int height, const int width, const int world_rank, const int world_size, MPI_Datatype MPI_Complex) 
{
    for (int i = begin; i < end; i++){
        fourierTransform(img + i * width, width, inverse, inverse);
    }

    if (world_rank != 0)
    {
        for (int i = begin; i < end; i++) 
        {
            MPI_Send(&img[width*i], width, MPI_Complex, 0, 0, MPI_COMM_WORLD);
        }
    }
    else 
    {
        for (int i = end; i < height; i++) 
        {
            const int rank = i*world_size/height;
            MPI_Recv(&img[width*i], width, MPI_Complex, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        transpose(img, width, height);
    }
}

int main(int argc, char** argv) {
    const bool inverse = argv[1][0] == 'r';

    InputImage image(argv[2]);
    
    const int height = image.get_height();
    const int width = image.get_width();
    const int imgSize = height * width;

    Complex* img = image.get_image_data();
    Complex imgOutput[imgSize];

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int array_of_blocklengths[] = { 1, 1 };
    MPI_Aint array_of_displacements[] = {offsetof(Complex, real), offsetof(Complex, imag)};
    MPI_Datatype array_of_types[] = {MPI_FLOAT, MPI_FLOAT};

    MPI_Datatype MPI_Complex;
    MPI_Type_create_struct(2,  
        array_of_blocklengths, 
        array_of_displacements,
        array_of_types,
        &MPI_Complex);
    MPI_Type_commit(&MPI_Complex);
    
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int begin = (height / world_size) * world_rank;
    const int end = begin + (height / world_size);

    gatherFourierTransform(img, inverse, begin, end, height, width, world_rank, world_size, MPI_Complex);

    if (world_rank == 0)
    {
        for (int i = end; i < height; i++) 
        {
            const int rank = i*world_size/width;
            MPI_Send(&img[width*i], width, MPI_Complex, rank, 0, MPI_COMM_WORLD);
        }
    }
    else 
    {
        for (int i = begin; i < end; i++) 
        {
            MPI_Recv(&img[width*i], width, MPI_Complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    gatherFourierTransform(img, inverse, begin, end, height, width, world_rank, world_size, MPI_Complex);

    if (world_rank == 0)
    {
        image.save_image_data(argv[3], img, width, height);
    }
    
    MPI_Finalize();

    return 0;
}