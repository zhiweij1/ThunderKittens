#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "testing_flags.dp.hpp"

#ifdef TEST_GANG

#include "testing_commons.dp.hpp"

namespace gang {

void tests(test_data &results);

}

#endif