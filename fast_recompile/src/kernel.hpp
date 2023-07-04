#include <sycl/sycl.hpp>
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
// #define d  0.5 * lamda
#define fifo_depth 4
// #define extended_length ChirpSize * ChirpSize
#define angleVecLenght 1801
constexpr int extended_length = ChirpSize * ChirpSize;
constexpr float lamda = lightSpeed / f0;
constexpr float d = 0.5 * lamda;

using preProcessingPipe = ext::intel::pipe<class preprocessingID, std::complex<float>, fifo_depth>;
using fft1dPipeArray = fpga_tools::PipeArray<class fft1dPipe, std::complex<float>, fifo_depth, RxSize, 1>;
using distAnglePipeArray = fpga_tools::PipeArray<class anglePipe, std::complex<float>, fifo_depth, RxSize, 1>;
using speedPipe = ext::intel::pipe<class speedPipeID, std::complex<float>, fifo_depth>;

/**
 * Host Function to submit the whole task producer 
*/
template<typename KernelClass, typename InPipe>
event SubmitHostProducer(queue &q, buffer<short,1> &base_frame, buffer<short,1> &raw_input)
{

    size_t num_elements = raw_input.size();
    std::cout << "Enqueuing Host Producer with size "<< num_elements << std::endl;
    auto e = q.submit([&](handler &h){
        accessor buf(raw_input, h, read_only);
        accessor base(base_frame, h, read_only);
        h.single_task<KernelClass>([=](){
            for(size_t i = 0; i < num_elements; i +=4)
            {   
                std::complex<float> tmp(buf[i] - base[i], buf[i+2] - base[i+2]);
                InPipe::write(tmp);
                tmp.real(buf[i+1] - base[i+1]);
                tmp.imag(buf[i+3] - base[i+3]);
                InPipe::write(tmp);
            }
        });
    });
    return e;
}

/**
 * preprocessing worker: reshape / pack the complex data, feed the data into later
 * 1-D fft kernel
*/
template<typename reshapeClass, typename partitionClass,typename InPipe>
event SubmitPreProcWorker(queue &q, buffer<std::complex<float>,1> reshaped_data)
{   
    std::cout << "Enqueuing preProc Worker" << std::endl;
    size_t num_elements = reshaped_data.size();
    auto e1 = q.submit([&](handler &h){
        accessor buf(reshaped_data, h, write_only);
        h.single_task<reshapeClass>([=](){
            for(size_t srcIdx = 0; srcIdx < num_elements; srcIdx ++)
            {
                // read num_elements / 4 * 2 times
                auto data = InPipe::read();
                int chirpIdx = srcIdx / (RxSize * SampleSize);
                int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;
                int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;
                int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;
                buf[destIdx] = data;
            }
        });
    });
    // e1.wait();
    
    auto e = q.submit([&](handler &h){
        accessor out(reshaped_data, h, read_only);
        h.depends_on(e1);
        h.single_task<partitionClass>([=](){
            fpga_tools::UnrolledLoop<RxSize>([&](auto id){
                int start = id * SampleSize * ChirpSize;
                int end = (id + 1) * SampleSize * ChirpSize;
                // size_t num_elements = SampleSize * ChirpSize;
                for(size_t idx = start; idx < end; idx ++)
                {
                    fft1dPipeArray::PipeAt<id,0>::write(out[idx]);
                }
            });
        });
    });

    return e;
} 

SYCL_EXTERNAL int kernelBitsReverse(int num, int bits);

SYCL_EXTERNAL extern "C" unsigned RtlByteswap(unsigned x);

SYCL_EXTERNAL float SyclSquare(float x);

/**
 * submit 1d fft worker
*/
template<typename PartitionKernelClass, typename FFTKernelClass,size_t id, size_t chunk_size, size_t num_elements>
event SubmitParitionFFTWorker(queue &q, buffer<std::complex<float>,1> partitioned_buffer, 
                             buffer<std::complex<float>,1> fftRes, int pow)
{
    auto e = q.submit([&](handler &h){
        accessor res(fftRes, h, write_only);
        accessor buf(partitioned_buffer,h,write_only);
        h.single_task<PartitionKernelClass>([=](){
            for(size_t i = 0; i < SampleSize * ChirpSize; i ++)
            {
                auto data =  fft1dPipeArray::PipeAt<id,0>::read();
                res[i] = data;
                buf[i] = data;
            }
        });
    });
    // e.wait();
    auto fftEvent = q.submit([&](handler &h){
        h.depends_on(e);
        accessor chunk(fftRes, h, read_write);
        h.single_task<FFTKernelClass>([=](){
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
                    chunk[i + offset].real(real(chunk[i+offset]) / chunk_size);
                    chunk[i + offset].imag(imag(chunk[i+offset]) / chunk_size);

                }
                // butterfly computation for each chunk 
                for(int mid = 1; mid < chunk_size; mid <<=1)
                {
                    std::complex<float> Wn(cos(PI / mid), -sin(PI/mid));
                    for(int stride = mid << 1, j = 0; j < chunk_size; j += stride)
                    {
                        std::complex<float> w(1,0);
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
 * Submit angle worker
*/
template <typename AngleWorkerKernelClass>
event SubmitAngleWorker(queue &q, buffer<std::complex<float>,1> &angle_weight, 
                            buffer<std::complex<float>,1> &phase_matrix,
                            buffer<std::complex<float>,1> &angle_vector)
{
    std::cout << "Enqueuing angle worker\n";
    // angle_weights (1 x 4) X phase_matrix (4 x 1801)
    auto angleWorker = q.submit([&](handler &h){
        accessor weight(angle_weight, h, read_only);
        accessor matrix(phase_matrix, h, read_only);
        accessor vector(angle_vector,h ,write_only);
        h.single_task<AngleWorkerKernelClass>([=]{
            for(int i = 0; i < angleVecLenght; i ++)
            {
                std::complex<float> res(0,0);
                for(int j = 0; j < RxSize; j++)
                {
                    auto weightData = weight[j];
                    auto matrixData = matrix[j * angleVecLenght + i];
                    res += (weightData * matrixData) ;
                }
                vector[i] = res;
            }
        });
    });
    return angleWorker;
}



template<typename SpeedProducerKernel, typename SpeedReceiverKernel, 
        typename SpeedTransposeKernel, typename SpeedFFTKernel1,
        typename SpeedFFTKernel2,typename SpeedFFTSwapKernel,
        size_t chunk_size, size_t num_elements>
event SubmitSpeedWorker(queue &q, buffer<std::complex<float>,1> rx0_raw_buffer, 
                        buffer<std::complex<float>,1> rx0_extend_buffer,
                        buffer<std::complex<float>,1> transpose_buffer, 
                        buffer<std::complex<float>,1> fftRes)
{
    std::cout << "Enqueuing Speed Worker\n";
    auto producerEvent = q.submit([&](handler &h){
        accessor rx0_raw(rx0_raw_buffer, h, read_only);
        h.single_task<SpeedProducerKernel>([=]{
            // total write SampleSize * ChirpSize times
            for(int i = 0; i < SampleSize * ChirpSize; i ++)
            {
                speedPipe::write(rx0_raw[i]);
            }
        });
    });

//     std::cout << "Enqueuing Speed Receiver\n";
    auto speedReceiverEvent = q.submit([&](handler &h){
        accessor rx0_extend(rx0_extend_buffer, h, write_only);
        accessor fft_res(fftRes, h, write_only);

        // h.depends_on(producerEvent);
        h.single_task<SpeedReceiverKernel>([=]{
            // total read SampleSize * ChirpSize times
            for(int i = 0; i < ChirpSize; i ++)
            {
                for(int j = 0; j < SampleSize; j ++)
                {
                    auto data = speedPipe::read();
                    int idx = i * ChirpSize + j;
                    rx0_extend[idx] = data;
                    fft_res[idx] = data;
                }
            }
        });
    });

    // std::cout << "Enqueuing speed FFT1 Worker\n";
    // q.wait();
    auto fftEvent1 = q.submit([&](handler &h){
        h.depends_on(speedReceiverEvent);
        accessor chunk(fftRes, h, read_write);
        h.single_task<SpeedFFTKernel1>([=](){
            fpga_tools::UnrolledLoop<num_elements / chunk_size>([&](auto chunk_idx){
                // bit reverse for each chunk
                size_t offset = chunk_idx * chunk_size;
                // auto chunk = out[offset];
                for(size_t i = 0; i < chunk_size; i++)
                {
                    int reversedIdx = kernelBitsReverse(i, 7);
                    if(reversedIdx > i)
                    {
                        auto temp = chunk[i + offset];
                        chunk[i + offset] = chunk[reversedIdx + offset];
                        chunk[reversedIdx + offset] = temp;
                    }
                    chunk[i + offset].real(real(chunk[i+offset]));
                    chunk[i + offset].imag(imag(chunk[i+offset]));

                }
                // butterfly computation for each chunk 
                for(int mid = 1; mid < chunk_size; mid <<=1)
                {
                    std::complex<float> Wn(cos(PI / mid), -sin(PI/mid));
                    for(int stride = mid << 1, j = 0; j < chunk_size; j += stride)
                    {
                        std::complex<float> w(1,0);
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

    std::cout << "Enqueuing speed transpose Worker\n";
    auto tranposeEvent = q.submit([&](handler &h){
        accessor transpose(transpose_buffer, h, write_only);
        accessor fft(fftRes, h, read_only);
        h.depends_on(fftEvent1);
        h.single_task<SpeedTransposeKernel>([=]{
            for(int col = 0; col < ChirpSize; col ++)
            {
                for(int row = 0; row < ChirpSize; row ++)
                {
                    int src = row * ChirpSize + col;
                    int dst = col * ChirpSize + row;
                    transpose[dst] = fft[src];
                }
            }
        });
    });

    std::cout << "Enqueuing speed FFT2 Worker\n";
    auto fftEvent2 = q.submit([&](handler &h){
        h.depends_on(tranposeEvent);
        accessor chunk(transpose_buffer, h, read_write);
        h.single_task<SpeedFFTKernel2>([=](){
            fpga_tools::UnrolledLoop<num_elements / chunk_size>([&](auto chunk_idx){
                // bit reverse for each chunk
                size_t offset = chunk_idx * chunk_size;
                // auto chunk = out[offset];
                for(size_t i = 0; i < chunk_size; i++)
                {
                    int reversedIdx = kernelBitsReverse(i, 7);
                    if(reversedIdx > i)
                    {
                        auto temp = chunk[i + offset];
                        chunk[i + offset] = chunk[reversedIdx + offset];
                        chunk[reversedIdx + offset] = temp;
                    }
                    chunk[i + offset].real(real(chunk[i+offset]));
                    chunk[i + offset].imag(imag(chunk[i+offset]));

                }
                // butterfly computation for each chunk 
                for(int mid = 1; mid < chunk_size; mid <<=1)
                {
                    std::complex<float> Wn(cos(PI / mid), -sin(PI/mid));
                    for(int stride = mid << 1, j = 0; j < chunk_size; j += stride)
                    {
                        std::complex<float> w(1,0);
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
    
    std::cout << "Swaping Left and Right FFT result\n";
    auto fftSwapEvent = q.submit([&](handler &h){
        h.depends_on(fftEvent2);
        accessor dst(fftRes, h, write_only);
        accessor src(transpose_buffer,h , read_only);
        h.single_task<SpeedFFTSwapKernel>([=](){
            fpga_tools::UnrolledLoop<num_elements / chunk_size>([&](auto idx){
                int start = idx * chunk_size;
                int end = start + chunk_size;
                int mid = start + chunk_size / 2;
                for(int srcIdx = start; srcIdx < end; srcIdx ++)
                {
                    int offset = (srcIdx < mid) ?  ((int)chunk_size / 2) : -((int)chunk_size / 2); 
                    int destIdx = srcIdx + offset;
                    if(destIdx == mid){
                        dst[destIdx] = std::complex<float>(0,0);
                    }else{
                        dst[destIdx]= src[srcIdx];
                    }

                }
            });
        });
    });

    return fftEvent1;
}


/**
 * fft helper function
*/
namespace fft_helper{
    int bitsReverse(int num, int bits);
    void hostFFT(std::vector<std::complex<float>> &input, int bits);
    void newFFT(std::vector<std::complex<float>> &x, int len);
}
