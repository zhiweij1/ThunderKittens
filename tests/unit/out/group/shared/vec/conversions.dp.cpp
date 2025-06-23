#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"

#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS

struct vec_copy {
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_vec_convert";
    template<int S, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template <int S, int NW, gl_t GL>
    SYCL_EXTERNAL static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        /*
        DPCT1115:67: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        auto &vec1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::col_vec<kittens::st_bf<16 * S, 16 * S>>>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        /*
        DPCT1115:68: The sycl::ext::oneapi::group_local_memory_for_overwrite is
        used to allocate group-local memory at the none kernel functor scope of
        a work-group data parallel kernel. You may need to adjust the code.
        */
        auto &vec2 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            kittens::col_vec<kittens::st_bf<16 * S, 16 * S>>>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        G::load(vec1, input, {});
        G::sync(0);
        G::copy(vec2, vec1);
        G::sync(0);
        G::store(output, vec2, {});
    }
};

void group::shared::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_copy, SIZE, 2>::run(results);
    sweep_size_1d<vec_copy, SIZE, 4>::run(results);
    sweep_size_1d<vec_copy, SIZE, 12>::run(results);
}

#endif