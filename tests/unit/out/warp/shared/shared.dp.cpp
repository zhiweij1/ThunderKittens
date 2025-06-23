#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared.dp.hpp"

#ifdef TEST_WARP_SHARED

void warp::shared::tests(test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/shared tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_TILE
    warp::shared::tile::tests(results);
#endif
#ifdef TEST_WARP_SHARED_VEC
    warp::shared::vec::tests(results);
#endif
}

#endif