#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GROUP_MEMORY_VEC_PGL_TO_SHARED

#include "testing_commons.dp.hpp"

namespace group {
namespace memory {
namespace vec {
namespace pgl_to_shared {

void tests(test_data &results);

}
}
}
}

#endif