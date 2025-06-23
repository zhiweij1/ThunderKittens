#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "warp.dp.hpp"

#ifdef TEST_WARP

using namespace warp;

void warp::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
#ifdef TEST_WARP_MEMORY
    memory::tests(results);
#endif
#ifdef TEST_WARP_REGISTER
    reg::tests(results); // register is a reserved word, hence reg
#endif
#ifdef TEST_WARP_SHARED
    shared::tests(results);
#endif
}

#endif