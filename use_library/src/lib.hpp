// =============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

// SYCL_EXTERNAL extern "C" unsigned RtlByteswap(unsigned x);

SYCL_EXTERNAL extern "C" unsigned RtlByteswap(unsigned x, unsigned y);
