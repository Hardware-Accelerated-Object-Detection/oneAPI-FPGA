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
int fft_helper::bitsReverse(int num, int bits){

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

void fft_helper::hostFFT(std::vector<std::complex<double>> &input, int bits)
{
    // std::cout << "Bit Reverse\n";
    int len = input.size();
    for(int i = 0; i < len ; i ++)
    {
        int reversedIdx = bitsReverse(i, bits);
        if(reversedIdx > i)
        {
            std::swap(input[i],input[reversedIdx]);
        }
    }
    // std::cout << "butterfly computaiton\n";
    for(int mid = 1; mid < len; mid <<=1)
    {
        std::complex<double> Wn(cos(PI / mid), -sin(PI / mid));
        for(int r = mid << 1, j = 0; j < len; j += r)
        {
            std::complex<double> w(1,0);
            for(int k = 0; k < mid; k++, w = w * Wn)
            {
                std::complex<double> a = input[j + k];
                std::complex<double> b = input[j + mid + k] * w;
                input[j + k] = a + b;
                input[j + mid + k] = a - b;
            }
        }
    }
}

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