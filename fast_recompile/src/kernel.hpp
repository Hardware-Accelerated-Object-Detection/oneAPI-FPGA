//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>
#include <mkl.h>
#include "../../include/pipe_utils.hpp"
#include "../../include/unrolled_loop.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
using namespace sycl;

#define SampleSize 100 // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128  // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 90   // the frame number
#define RxSize 4       // the rx size, which is usually 4
#define numTx 1
#define numRx 4 // the rx size, which is usually 4
#define THREADS_PER_BLOCK 512
#define PI 3.14159265359 // pi
#define fs 2.0e6         // sampling frequency
#define lightSpeed 3.0e08
#define mu 5.987e12 // FM slope
#define f0 77e9
#define lamda lightSpeed / f0
#define d = 0.5 * lamda
#define fifo_depth 4
#define extended_length ChirpSize * ChirpSize


using preProcessingPipe = ext::intel::pipe<class preprocessingID, std::complex<double>, fifo_depth>;
class preprocessingProducerClass;
class preprocessingConsumerClass;

event PreProcessingProducer(queue &q, buffer<short,1> &base_frame, buffer<short,1> &output);
// event PreProcessingConsumer(queue &q, buffer<std::complex<double>,1> &output, std::array<std::vector<std::complex<double>>, RxSize> &partitioned_data);
event PreProcessingConsumer(queue &q, buffer<std::complex<double>,1> &output);

using fftPipeArray = fpga_tools::PipeArray<
    class fftPipeID, std::complex<double>,
    fifo_depth, RxSize>;

/**
 * fft producer & consumer.
*/



template<size_t producer_id> class fftProducerClass;
template <size_t consumer_id, size_t chunk_size, size_t num_elements> class fftConsumerReadClass;
template <size_t consumer_id, size_t chunk_size, size_t num_elements> class fftConsumerButerflyClass;


template<size_t producer_id>
event fftProducer(queue &q, buffer<std::complex<double>,1> &input)
{
    std::cout << "Enqueuing FFT Producer " << producer_id << std::endl;
    auto e = q.submit([&](handler &h){
        accessor in(input, h, read_only);
        auto num_elements = input.size();
        h.single_task<fftProducerClass<producer_id>>([=](){
            size_t i = 0;
            for(size_t pass = 0; pass < num_elements; pass ++){
                fftPipeArray::PipeAt<producer_id>::write(in[i++]);
            }
        });
    });
    return e;
}


/**
 * fft helper function
*/
namespace fft_helper{
    int bitsReverse(int num, int bits);
    void hostFFT(std::vector<std::complex<double>> &input, int bits);
}


SYCL_EXTERNAL int kernelBitsReverse(int num, int bits);


template <size_t consumer_id, size_t chunk_size, size_t num_elements>
event fftConsumer(queue &q, buffer<std::complex<double>,1> &output, int pow)
{
    std::cout << "Enqueuing FFT Consumer " << consumer_id << std::endl;
    auto dataReadEvent = q.submit([&](handler &h){
        accessor out(output, h, write_only);
        h.single_task<fftConsumerReadClass<consumer_id, chunk_size, num_elements>>([=](){
            for(size_t i = 0; i < num_elements; i ++)
            {
                out[i] = fftPipeArray::PipeAt<consumer_id>::read();
            }
        });
    });
    dataReadEvent.wait();
    auto fftEvent = q.submit([&](handler &h){
        accessor chunk(output, h, read_write);
        h.single_task<fftConsumerButerflyClass<consumer_id,chunk_size, num_elements>>([=](){
            fpga_tools::UnrolledLoop<num_elements / chunk_size>([&](auto chunk_idx){
                // bit reverse for each chunk
                size_t offset = chunk_idx * chunk_size;
                // auto chunk = out[offset];
                for(size_t i = 0; i < chunk_size; i++)
                {
                    int reversedIdx = kernelBitsReverse(i, pow);
                    if(reversedIdx > i)
                    {
                        auto temp = chunk[i + offset];
                        chunk[i + offset] = chunk[reversedIdx + offset];
                        chunk[reversedIdx + offset] = temp;
                    }
                }
                // butterfly computation for each chunk 
                for(int mid = 1; mid < chunk_size; mid <<=1)
                {
                    std::complex<double> Wn(cos(PI / mid), -sin(PI/mid));
                    for(int stride = mid << 1, j = 0; j < chunk_size; j += stride)
                    {
                        std::complex<double> w(1,0);
                        for(int k = 0; k < mid; k ++, w = w * Wn)
                        {
                            auto a = chunk[j + k + offset];
                            auto b = chunk[j + mid + k + offset] * w;
                            chunk[j + k + offset] = a + b;
                            chunk[j + mid + k + offset] = a - b;
                        }
                    }
                }
            
            });

        });
    });
    return fftEvent;
}   


/**
 * producer & consumer for speed detection.
*/
class speedDetectionClass;
template<size_t producer_id>
event speedDetectionProducer(queue &q, buffer<std::complex<double>,1> &input);

class speedDetectionClass;
template<size_t consumer_id>
event speedDetectionConsumer(queue &q, buffer<std::complex<double>,1> &output);
