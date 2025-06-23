#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "vec.dp.hpp"

#ifdef TEST_GROUP_SHARED_VEC

void group::shared::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
    group::shared::vec::conversions::tests(results);
#endif
#ifdef TEST_GROUP_SHARED_VEC_MAPS
    group::shared::vec::maps::tests(results);
#endif
}

#endif