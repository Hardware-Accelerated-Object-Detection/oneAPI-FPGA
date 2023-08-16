//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <iostream>
SYCL_EXTERNAL extern "C" unsigned RtlByteswap (unsigned x, unsigned y) {
  // return (x << 16) | (y >> 16);
  return x + y;
}

 