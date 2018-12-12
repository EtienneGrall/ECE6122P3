#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <stdio.h>

#include "complex.h"
#include "input_image.cc"

using namespace std;

#define NB_THREADS 8
#define PI 3.14159265358979f

Complex W(const int exponent, const int N)
{
	return Complex(cos(2*PI*exponent/N), -sin(2*PI*exponent/N));
}

void DFT1D(Complex *img, Complex *imgTransformed, const int firstRow, const int lastRow, const float N, const bool row, const bool inverse)
{
	const float coef = inverse ? -1 : 1;
	for (int r = firstRow; r < lastRow; r++)
	{
		for (int i = 0; i < N; i++)
		{
			const int indexOriginal = row ? N*r + i : N*i + r;
			for (int k = 0; k < N; k++)
			{
				const int indexTransformed = row ? N*r + k : N*k + r;
				imgTransformed[indexOriginal] = imgTransformed[indexOriginal] + img[indexTransformed]*W(coef*i*k, N);
			}
			imgTransformed[indexOriginal] = imgTransformed[indexOriginal] * ((inverse) ? Complex(1/N) : 1);
		}
	}
}

vector<int> init(const int N)
{
	vector<int> nbPerThreads;
	if (N < NB_THREADS) 
	{
		for (int i = 0; i < N; i++)
		{
			nbPerThreads.push_back(i+1);
		}
		return nbPerThreads;
	}

	for (int i = 0; i < NB_THREADS; i++)
	{
		nbPerThreads.push_back(N/NB_THREADS);
		if (i > 0)
		{
			nbPerThreads[i] += nbPerThreads[i-1];
		}
		if (i < (N%NB_THREADS))
		{
			nbPerThreads[i]++;
		}
	}
	return nbPerThreads;
	
}

int main (int argc, char** argv)
{
	const bool inverse = argv[1][0] == 'r';
	InputImage image(argv[2]);

	const int h = image.get_height();
	const int w = image.get_width();
    const int imgSize = h * w;

    Complex* img = image.get_image_data();

    Complex *imgTransformedRow; 
	imgTransformedRow = (Complex*)malloc(sizeof(Complex) * imgSize);
	for (int i = 0; i < imgSize; i++)
	{
		imgTransformedRow[i] = Complex();
	}

	vector<int> nbRowsThread = init(h);
    vector<thread> threads;

	for (int i = 0; i < nbRowsThread.size(); i++)
	{
			const int begin = (i > 0) ? nbRowsThread[i-1] : 0;
			const int end = nbRowsThread[i];

			thread thr(DFT1D, img, imgTransformedRow, begin, end, h, true, inverse);
			threads.push_back(move(thr));	
	}

	for (thread &thr : threads)
	{
		thr.join();
	}

	threads.clear();
	vector<int> nbColThread = init(w);

	Complex *imgTransformedCol; 
	imgTransformedCol = (Complex*)malloc(sizeof(Complex) * imgSize);
	for (int i = 0; i < imgSize; i++)
	{
		imgTransformedCol[i] = Complex();
	}

	for (int i = 0; i < nbColThread.size(); i++)
	{
			const int begin = (i > 0) ? nbColThread[i-1] : 0;
			const int end = nbColThread[i];

			thread thr(DFT1D, imgTransformedRow, imgTransformedCol, begin, end, w, false, inverse);
			threads.push_back(move(thr));	
	}

	for (thread &thr : threads)
	{
		thr.join();
	}
	
	image.save_image_data(argv[3], imgTransformedCol, w, h);
    return 0;
}