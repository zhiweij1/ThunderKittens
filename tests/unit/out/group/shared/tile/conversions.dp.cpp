#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conversions.dp.hpp"

#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS

struct test_shared_copy {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_copy";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template <int H, int W, int NW, gl_t GL>
    SYCL_EXTERNAL static void device_func(const GL &input, const GL &output,
                                          uint8_t *dpct_local) {
        using G = kittens::group<NW>;
        auto __shm = (kittens::alignment_dummy *)
            dpct_local; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<16*H, 16*W> &t1 = al.allocate<kittens::st_bf<16*H, 16*W>>();
        kittens::st_bf<16*H, 16*W> &t2 = al.allocate<kittens::st_bf<16*H, 16*W>>();
        G::load(t2, input, {});
        G::sync(0);
        G::copy(t1, t2);
        G::sync(0);
        G::store(output, t1, {});
    }
};

void group::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<test_shared_copy, SIZE, SIZE, 2>::run(results);

    if constexpr (TEST_INTENSITY > 1) {

        sweep_size_2d<test_shared_copy, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_size_2d<test_shared_copy, 12, 4, 12>::run(results);

        }
    }
}

#endif