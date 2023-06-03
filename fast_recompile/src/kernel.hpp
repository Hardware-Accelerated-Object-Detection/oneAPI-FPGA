//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
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

class Complex_t
{
public:
    double real;
    double imag;
    /**
     * Class constructor
     */
    Complex_t(double r, double i)
    {
        real = r;
        imag = i;
    }

    Complex_t()
    {
        real = 0.0;
        imag = 0.0;
    }
    /**
     * operator overloading
     */
    Complex_t operator+(const Complex_t &a)
    {
        Complex_t res;
        res.real = a.real + real;
        res.imag = a.imag + imag;
        return res;
    }
    Complex_t operator-(const Complex_t &a)
    {
        Complex_t res;
        res.real = real - a.real;
        res.imag = imag - a.imag;
        return res;
    }
    Complex_t operator*(const Complex_t &a)
    {
        Complex_t res;
        res.real = a.real * real - a.imag * imag;
        res.imag = a.real * imag - a.imag * real;
        return res;
    }
    bool operator!=(const Complex_t &a)
    {
        return (real == a.real && imag == a.imag);
    }
    bool operator==(const Complex_t &a)
    {
        return (real == a.real && imag == a.imag);
    }

    double getModulu()
    {
        return (sqrt(real * real + imag * imag));
    }
};
// initialize the reading to preprocessing pipe in FPGA
using ReadToReshapePipe = ext::intel::pipe<class ReadToPreprocPipeID, short, fifo_depth>;
class ReadProducerClass;
class ReshapeConsumerClass;
/**
 * @param ReadProducer: pack the input short typed raw data into complex type a
 * and transmitt the complex typed data to producer for reshape and pairing
 * @param q: device queue.
 * @param raw_buffer: buffer stores the raw short typed data.
 */
event ReadProducer(queue &q, buffer<short, 1> &raw_buffer);
/**
 * reshape consumer event
 * @param q: device queue.
 * @param hold_buf: buffer to hold the raw data from fifo with depth = fifo_depth;
 * @param out_buf: output buffer filled with reshaped data with depth = RxSize * ChirpSize * SampleSize / 2;
 * 
*/
event ReshapeConsumer(queue &q, buffer<short,1> &hold_buf, buffer<Complex_t, 1> &out_buf);
