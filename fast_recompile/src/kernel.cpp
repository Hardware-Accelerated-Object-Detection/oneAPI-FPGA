//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"


/**
 * Read raw data kernel function
 * @param q: device queue.
 * @param raw_buffer: buffer stores the raw short typed data.
 * @return event 
 */
event ReadProducer(queue &q, buffer<short, 1> &raw_buffer)
{
    std::cout << "Packing the raw data into Complex form...\n"
              << "Enqueuing ReadProducer pipe...\n";
    auto e = q.submit([&](handler &h)
                      {
        accessor raw_data_accessor(raw_buffer,h,read_only);
        size_t num_elements = raw_buffer.size();
            // enqueue the short raw data into pipe
        h.single_task<ReadProducerClass>([=](){
            // buffer<short,1>*ptr = raw_buffer;
        for(size_t i = 0; i < num_elements; i++){
                // temp.real=(double)raw_buffer[i];
                // temp.imag=(double)raw_buffer[i+2];
               ReadToReshapePipe::write(raw_data_accessor[i]);
            }
            }); });
            return e;
}


/**
 * reshape consumer kernel function.
 * @param q: device queue.
 * @param hold_buf: buffer to hold the raw data from fifo with depth = fifo_depth;
 * @param out_buf: output buffer filled with reshaped data with depth = RxSize * ChirpSize * SampleSize / 2;
 * @return event
*/
event ReshapeConsumer(queue &q, buffer<short,1> &hold_buf, buffer<Complex_t, 1> &out_buf)
{
    std::cout << "Reshaping raw data into complex form ...\n"
              << "Enqueuing ReshapeConsumerPipe...\n";
    // read from the pipe and pack 2 data together into complex form
    auto e = q.submit([&](handler &h)
    {
        // auto out_accessor = out_buf.get_access<access::mode::write>(h);
        // auto hold_accessor = hold_buf.get_access<access::mode:read_write>(h);
        accessor out_accessor(out_buf, h, write_only);
        accessor hold_accessor(hold_buf, h, read_write);
        size_t num_elements = out_buf.size() * 2;

        h.single_task<ReshapeConsumerClass>([=](){
            for(size_t i = 0; i < num_elements; i += 4){
                // read the 'short' data from pipe
                for(size_t j = 0; j < fifo_depth; j +=4){
                    hold_accessor[j] = ReadToReshapePipe::read();
                    hold_accessor[j + 1] = ReadToReshapePipe::read();
                    hold_accessor[j + 2] = ReadToReshapePipe::read();
                    hold_accessor[j + 3] = ReadToReshapePipe::read();
                }
                // corresponding index in complex form array
                int srcIdx = i / 2;
                // pack the data from pipe into Complex_t
                for(int k = 0; k < fifo_depth/2; k++){
                    
                    Complex_t temp = Complex_t(hold_accessor[k],hold_accessor[k+2]);
                    // calculate destination index
                    int chirpIdx = srcIdx / (RxSize * SampleSize);
                    int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;
                    int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;
                    int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;
                    out_accessor[destIdx] = temp;
                    srcIdx ++;
                }

            }
        });
    });
  return e;
}
