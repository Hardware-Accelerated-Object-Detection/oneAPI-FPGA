echo "genrating rtl_lib..."
fpga_crossgen ../src/lib_rtl_spec.xml --emulation_model ../src/lib_rtl_model.cpp --target sycl -o lib_rtl.o

echo "generating lib.a..."
fpga_libtool lib_rtl.o --target sycl --create lib.a

echo "compiling for fpga_simulation..."
icpx -fsycl -fintelfpga ../src/use_library.cpp lib.a -o fpga.sim -Xssimulation -Xsghdl=0 -DFPGA_SIMULATOR=1

# icpx -fsycl -fintelfpga ../src/use_library.cpp lib.a -o use_library.fpga_sim -Xssimulation -DFPGA_SIMULATOR=1
