#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_WGMMA

#include "testing_commons.cuh"

#include "mma_fp32_fp8.cuh"
#include "mma_fp16_fp8.cuh"
#include "mma_fp32_bf16.cuh"
#include "mma_fp32_fp16.cuh"
#include "mma_fp16_fp16.cuh"
// #include "mma_fp32_fp32.cuh" TODO
#include "complex/complex_wgmma.cuh"

namespace group {
namespace wgmma {

void tests(test_data &results);

}
}

#endif