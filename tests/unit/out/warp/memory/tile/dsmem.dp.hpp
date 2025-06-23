#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE_DSMEM

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace tile {
namespace dsmem {

void tests(test_data &results);

}
}
}
}

#endif