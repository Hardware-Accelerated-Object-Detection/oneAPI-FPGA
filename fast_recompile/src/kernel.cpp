//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"

event PreProcessingProducer(queue &q, buffer<short,1> &base_frame, buffer<short,1> &raw_input)
{
    std::cout << "Enqueuing test Consumer" << std::endl;
    size_t num_elements = raw_input.size();
    auto e = q.submit([&](handler &h){
        accessor buf(raw_input, h, read_only);
        accessor base(base_frame, h, read_only);
        h.single_task([=](){
            for(size_t i = 0; i < num_elements; i +=4)
            {   
                std::complex<double> tmp(buf[i] - base[i], buf[i+2] - base[i+2]);
                // tmp.real(real(buf[i]) + real(buf[i+2]));
                testPipe::write(tmp);
                // tmp = buf[i + 1] + buf[i+3];
                tmp.real(buf[i+1] - base[i+1]);
                tmp.imag(buf[i+3] - base[i+3]);
                // tmp.imag(imag(buf[i+1])+imag(buf[i+3]));
                testPipe::write(tmp);
            }
        });
    });

    return e;
}


event PreProcessingConsumer(queue &q, buffer<std::complex<double>,1> &output)
{
    std::cout << "Enqueuing test Consumer" << std::endl;
    size_t num_elements = output.size();
    auto e = q.submit([&](handler &h){
        accessor buf(output, h, write_only);
        h.single_task([=](){
            for(size_t srcIdx = 0; srcIdx < num_elements; srcIdx ++)
            {
                // read num_elements / 4 * 2 times
                auto data = testPipe::read();
                int chirpIdx = srcIdx / (RxSize * SampleSize);
                int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;
                int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;
                int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;
                buf[destIdx] = data;
                // data = testPipe::read();
                // buf[i+1] = data;
            }
        });
    });
    return e;
}

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