//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"

/**
 * fft helper function
 */

int length2Pow(int len) {
  int cnt = 0;
  int tmp = 1;
  while (tmp < len) {
    tmp <<= 1;
    cnt++;
  }
  return cnt;
}
/**
 * fft helper function to reverse index bits for later butterfly compution.
 * @param num input index number.
 * @param bits total number of bits of size of array = length2pow(array.size()). 
*/
int fft_helper::bitsReverse(int num, int bits)
{

    int rev = 0;
    for (int i = 0; i < bits; i++)
    {
        if (num & (1 << i))
        {
            rev |= 1 << ((bits - 1) - i);
        }
    }
    return rev;
}
/**
 * host version of fft
*/
void fft_helper::hostFFT(std::vector<std::complex<float>> &input, int bits)
{
    // std::cout << "Bit Reverse\n";
    int len = input.size();
    for (int i = 0; i < len; i++)
    {
        int reversedIdx = bitsReverse(i, bits);
        if (reversedIdx > i)
        {
            std::swap(input[i], input[reversedIdx]);
        }
    }
    // std::cout << "butterfly computaiton\n";
    for (int mid = 1; mid < len; mid <<= 1)
    {
        std::complex<float> Wn(cos(PI / mid), -sin(PI / mid));
        for (int r = mid << 1, j = 0; j < len; j += r)
        {
            std::complex<float> w(1, 0);
            for (int k = 0; k < mid; k++, w = w * Wn)
            {
                std::complex<float> a = input[j + k];
                std::complex<float> b = input[j + mid + k] * w;
                input[j + k] = a + b;
                input[j + mid + k] = a - b;
            }
        }
    }
}
/**
 * test version of fft in host for reference
*/
void fft_helper::newFFT(std::vector<std::complex<float>>&x, int len)
{
    int temp = 1, l = 0;
    int *r = (int *)malloc(sizeof(int) * len);
    // FFT index reverse，get the new index，l=log2(len)
    while (temp < len)
        temp <<= 1, l++;
     for (int i = 0; i < len; i++)
    {
        int reversedIdx = bitsReverse(i, l);
        if (reversedIdx > i)
        {
            std::swap(x[i], x[reversedIdx]);
        }
    }    
    for (int mid = 1; mid < len; mid <<= 1)
    {
        std::complex<float>Wn(cos(PI / mid), -sin(PI / mid)); /*drop the "-" sin，then divided by len to get the IFFT*/
        for (int R = mid << 1, j = 0; j < len; j += R)
        {
            std::complex<float>w(1, 0);
            for (int k = 0; k < mid; k++, w*= Wn)
            {
                std::complex<float> a = x[j + k];
                std::complex<float> b = w * x[j + mid + k];
                x[j + k] = a + b;
                x[j + mid + k] = a - b;
            }
        }
    }
    free(r);
}

/**
 * SYCL externel function for bit reverse
 * @param num input index number.
 * @param bits total number of bits of size of array = length2pow(array.size()). 
*/
SYCL_EXTERNAL int kernelBitsReverse(int num, int bits)
{
    int rev = 0;
    for (int i = 0; i < bits; i++)
    {
        if (num & (1 << i))
        {
            rev |= 1 << ((bits - 1) - i);
        }
    }
    return rev;
}

/**
 * SYCL externel function for RTL porting test.
 * 
*/
SYCL_EXTERNAL float SyclSquare(float x) {
  return x * x;
}

