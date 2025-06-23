#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_WGMMA_MMA_FP16_FP8

#include "testing_commons.cuh"

namespace group {
namespace wgmma {
namespace mma_fp16_fp8 {
void tests(test_data &results);
}
}
}

#endif
