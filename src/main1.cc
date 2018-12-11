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
	return Complex(cos(2*PI*exponent/N), -1*sin(2*PI*exponent/N));
}

void DFTRow(Complex *img, Complex *imgTransformed, const int firstRow, const int lastRow, const int N)
{
	for (int r = firstRow; r < lastRow; r++)
	{
		for (int i = 0; i < N; i++)
		{
			for (int k = 0; k < N; k++)
			{
				imgTransformed[N*r + i] = imgTransformed[N*r + i] + img[N*r + k]*W(i*k, N);
			}
		}
	}
}

void DFTCol(Complex *img, Complex *imgTransformed, const int firstCol, const int lastCol, const int N)
{
	for (int r = firstCol; r < lastCol; r++)
	{
		for (int i = 0; i < N; i++)
		{
			for (int k = 0; k < N; k++)
			{
				imgTransformed[N*i + r] = imgTransformed[N*i + r] + img[N*k + r]*W(i*k, N);
			}
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
	InputImage image(argv[2]);

	const int h = image.get_height();
	const int w = image.get_width();
    const int imgSize = h * w;

    Complex* img = image.get_image_data();

    Complex *imgTransformed; 
	imgTransformed = (Complex*)malloc(sizeof(Complex) * imgSize);
	for (int i = 0; i < imgSize; i++)
	{
		imgTransformed[i] = Complex();
	}

	vector<int> nbRowsThread = init(h);
    vector<thread> threads;

	for (int i = 0; i < nbRowsThread.size(); i++)
	{
			const int begin = (i > 0) ? nbRowsThread[i-1] : 0;
			const int end = nbRowsThread[i];

			thread thr(DFTRow, img, imgTransformed, begin, end, h);
			threads.push_back(move(thr));	
	}

	for (thread &thr : threads)
	{
		thr.join();
	}

	threads.clear();
	vector<int> nbColThread = init(w);

	for (int i = 0; i < nbColThread.size(); i++)
	{
			const int begin = (i > 0) ? nbColThread[i-1] : 0;
			const int end = nbColThread[i];

			thread thr(DFTCol, img, imgTransformed, begin, end, w);
			threads.push_back(move(thr));	
	}

	for (thread &thr : threads)
	{
		thr.join();
	}
	
	image.save_image_data(argv[3], imgTransformed, w, h);
    return 0;
}