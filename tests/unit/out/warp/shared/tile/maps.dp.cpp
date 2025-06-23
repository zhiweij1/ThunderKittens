#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "maps.dp.hpp"
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#ifdef TEST_WARP_SHARED_TILE_MAPS

template<typename T>
struct test_exp {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<( NW == 1 && W*H<=64 && sizeof(dtype) != 1 )>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_exp_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_exp_gmem=half" :
                                                                                         "shared_exp_gmem=float";
    template<int H, int W, int NW, gl_t GL> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < i_ref.size(); i++)
            o_ref[i] = sycl::ext::intel::math::bfloat162float(
                sycl::ext::intel::math::float2bfloat16(
                    ::expf(i_ref[i]))); // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL> static void device_func(const GL &input, const GL &output,
                                                                    uint8_t *dpct_local) {
        auto __shm = (kittens::alignment_dummy *)dpct_local;
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::load(shared_tile, input, {});
        kittens::exp(shared_tile, shared_tile);
        kittens::store(output, shared_tile, {});
    }
};

void warp::shared::tile::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_2d_warp<test_exp, SIZE, SIZE>::run(results);
}

#endif