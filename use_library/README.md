# Using FPGA Cross-Language Libraries
This library is modified from intel official oneapi sample which can be found at [use_library](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL_FPGA/Tutorials/Tools/use_library).

I use different compile command to compile the simulation result as I have different environment and found the CMake command given by Intel is always failing. If you want to use, do follows:
```bash
# create build directory
mkdir build
cd build
cmake ..
# move the paired compiling commands bash file to build directory
cp ../build_simulation.sh .
# compile for simulation
sh build_simulation.sh
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
