//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"

event PreProcessingProducer(queue &q, buffer<short, 1>& raw_input, buffer<std::complex<double>,1>& hold, buffer<short,1> &base_frame ,buffer<int,1> &write_time)
{
    std::cout << "PreProcessing producer \n";
    auto e = q.submit([&](handler &h){
        accessor raw_data(raw_input, h, read_only);
        accessor hold_buffer(hold, h, read_write);
        accessor write_buffer(write_time, h, read_write);
        accessor base_buffer(base_frame, h, read_only);
        size_t num_elements = raw_input.size();
        h.single_task<preprocessingProducerClass>([=](){
            int ptr = 0;
            for(int i = 0; i < num_elements; i += 4)
            {
                // total write times: SampleSize * ChirpleSize * Rx
                // pack the complex numebr 
                hold_buffer[ptr].real(raw_data[i] - base_buffer[i]);
                hold_buffer[ptr].imag(raw_data[i+2] - base_buffer[i+2]);
                preProcessingPipe::write(hold_buffer[ptr]);
                write_buffer[0] ++;
                ptr ++;
                hold_buffer[ptr].real(raw_data[i+1] - base_buffer[i + 1]);
                hold_buffer[ptr].imag(raw_data[i+3] - base_buffer[i + 3]);
                preProcessingPipe::write(hold_buffer[ptr]);
                write_buffer[0] ++;
                ptr++;
            }
        });
    });
    return e;
}


event PreProcessingConsumer(queue &q, buffer<std::complex<double>,1> &output, buffer<int,1> &read_time)
{
    std::cout << "preprocesing consumer\n";
    auto e = q.submit([&](handler &h){
        accessor out(output, h, write_only);
        accessor read_buffer(read_time, h, read_write);
        size_t num_elements = output.size();
        h.single_task<preprocessingConsumerClass>([=](){
            for(size_t srcIdx = 0; srcIdx < num_elements; srcIdx ++){
                int chirpIdx = srcIdx / (RxSize * SampleSize);
                int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;
                int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;
                int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;
                // total read time Sample * ChirpSize * RxSize
                auto data = preProcessingPipe::read();
                read_buffer[0] ++;
                out[destIdx] = data;
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