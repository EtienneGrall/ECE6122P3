#include <iostream>
#include <cmath>
#include <vector>
#include <thread>  

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
			//imgTransformed[N*r + i] = Complex term();
			for (int k = 0; k < N; k++)
			{
				imgTransformed[N*r + i] = imgTransformed[N*r + i] + img[i+k]*W(i*k, N);
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
			//imgTransformed[N*r + i] = Complex term();
			for (int k = 0; k < N; k++)
			{
				imgTransformed[N*r + i] = imgTransformed[N*r + i] + img[i+k]*W(i*k, N);
			}
		}
	}
}

int main (int argc, char** argv)
{
	cout << argv[2] << endl;
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

    vector<int> nbRowsThread;
	if (h < NB_THREADS) 
	{
		for (int i = 0; i < h; i++)
		{
			nbRowsThread.push_back(i+1);
		}
	}
	else
	{
		for (int i = 0; i < NB_THREADS; i++)
		{
			nbRowsThread.push_back(h/NB_THREADS);
			if (i > 0)
			{
				nbRowsThread[i] += nbRowsThread[i-1];
			}
			if (i < (h%NB_THREADS))
			{
				nbRowsThread[i]++;
			}
		}
	}

    vector<thread> threads;
	for (int i = 0; i < nbRowsThread.size(); i++)
	{
			const int begin = (i > 0) ? nbRowsThread[i-1] : 0;
			const int end = nbRowsThread[i];

			//for (int r = begin; r < end; r++)
			{
			thread thr(DFTRow, img, imgTransformed, begin, end, w);
			threads.push_back(move(thr));	
			}
	}

	for (thread &thr : threads)
	{
		thr.join();
	}

	for (int i = 0; i < nbRowsThread.size(); i++)
	{
			const int begin = (i > 0) ? nbRowsThread[i-1] : 0;
			const int end = nbRowsThread[i];

			//for (int r = begin; r < end; r++)
			{
			thread thr(DFTCol, img, imgTransformed, begin, end, w);
			threads.push_back(move(thr));	
			}
	}

	for (thread &thr : threads)
	{
		thr.join();
	}

	image.save_image_data(argv[3], imgTransformed, w, h);
    return 0;
}