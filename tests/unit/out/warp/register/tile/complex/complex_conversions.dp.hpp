#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS_COMPLEX

#include "testing_commons.cuh"

namespace warp {
namespace reg {
namespace tile {
namespace complex {
namespace conversions {

void tests(test_data &results);

}
}
}
}
}

#endif