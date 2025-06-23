#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "reductions.dp.hpp"
#include <sycl/ext/intel/math.hpp>

#include <cmath>

#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS

template<typename T>
struct vec_norm {
    using dtype = T;
    template<int S, int NW>
    using valid = std::bool_constant<NW == 1 && S<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_norm_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_norm_gmem=half" :
                                                                                         "shared_vec_norm_gmem=float";
    template<int S, int NW, gl_t GL>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // turns out to get the numerics right in bf16 you have to actually simulate the reduction tree :/
        kittens::bf16 sum[32] = sycl::ext::intel::math::float2bfloat16(0.f);
        if constexpr (S > 1) {
            for (int i = 0; i < 32; i++)
                sum[i] = sycl::ext::intel::math::float2bfloat16(abs(i_ref[i]));
            for (int i = 32; i < o_ref.size(); i++) sum[i % 32] +=
                sycl::ext::intel::math::float2bfloat16(abs(i_ref[i]));
            // now reduce first step
            for(int i = 0; i < 16; i++) sum[i] += sum[i+16];
        }
        else {
            for (int i = 0; i < 16; i++)
                sum[i] = sycl::ext::intel::math::float2bfloat16(abs(i_ref[i]));
        }
        for(int i = 0; i < 8; i++) sum[i] += sum[i+8];
        for(int i = 0; i < 4; i++) sum[i] += sum[i+4];
        for(int i = 0; i < 2; i++) sum[i] += sum[i+2];
        sum[0] += sum[1];
        sum[0] += sycl::ext::intel::math::float2bfloat16(1.f);
        for(int i = 0; i < o_ref.size(); i++) {
            kittens::bf16 o =
                sycl::ext::intel::math::float2bfloat16(i_ref[i]) / sum[0];
            o_ref[i] = sycl::ext::intel::math::bfloat162float(o);
        }
    }
    template<int S, int NW, gl_t GL>
    static void device_func(const GL &input, const GL &output,
                            uint8_t *dpct_local) {
        auto __shm = (kittens::alignment_dummy *)dpct_local;
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &vec    = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &absvec = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        kittens::load(vec, input, {});
        kittens::abs(absvec, vec);
        dtype f = kittens::base_types::constants<dtype>::one();
        kittens::sum(f, absvec, f);
        kittens::div(vec, vec, f);
        kittens::store(output, vec, {});
    }
};

void warp::shared::vec::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_1d_warp<vec_norm, SIZE>::run(results);
}

#endif