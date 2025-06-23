#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "global_to_shared.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static void g2s_global_wrapper_2d(const GL input, const GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename axis, typename... args>
struct g2s_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, axis, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, axis, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int SIZE = H*W*B*D*R*C*256;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, -1, -1, 16*C*W>;
            static_assert(axis::value==0 || axis::value==1 || axis::value==2, "Axis must be 0, 1, or 2.");
            GL input  (d_i, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            GL output (d_o, (axis::value==0?H*16:1)*B, (axis::value==1?H*16:1)*D, (axis::value==2?H*16:1)*R, nullptr);
            // run kernel
            /*
            DPCT1026:58: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
            /*
            DPCT1049:12: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
                  {
                        auto exp_props =
                            sycl::ext::oneapi::experimental::properties{
                                sycl::ext::oneapi::experimental::use_root_sync};

                        dpct::get_in_order_queue().submit([&](sycl::handler
                                                                  &cgh) {
                              cgh.depends_on(
                                  dpct::get_current_device()
                                      .get_in_order_queues_last_events());

                              cgh.parallel_for(
                                  sycl::nd_range<3>(
                                      sycl::range<3>(1, 1, NUM_WORKERS * 32),
                                      sycl::range<3>(1, 1, NUM_WORKERS * 32)),
                                  exp_props, [=](sycl::nd_item<3> item_ct1) {
                                        global_wrapper_2d<test, dtype, H, W,
                                                          NUM_WORKERS, GL, axis,
                                                          args...>(input,
                                                                   output);
                                  });
                        });
                  }
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, axis, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2s_sweep_size_2d = loop_h<g2s_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using g2s_sweep_size_2d_warp = g2s_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args>
struct g2s_sweep_gmem_type_2d_warp {
    static void run(test_data &results) {
        g2s_sweep_size_2d_warp<test<float>, MAX_H, MAX_W, args...>::run(results);
        g2s_sweep_size_2d_warp<test<kittens::bf16>, MAX_H, MAX_W, args...>::run(results);
        g2s_sweep_size_2d_warp<test<kittens::half>, MAX_H, MAX_W, args...>::run(results);
        #ifdef KITTENS_HOPPER
        g2s_sweep_size_2d_warp<test<kittens::fp8e4m3>, MAX_H, MAX_W, args...>::run(results);
        g2s_sweep_size_2d_warp<test<kittens::fp8e5m2>, MAX_H, MAX_W, args...>::run(results);
        #endif
    }
};


template<typename T>
struct st_load_store {
    using dtype = T;
    template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64) 
        #ifdef KITTENS_HOPPER
        && ( ( !std::is_same_v<kittens::fp8e4m3, T> && !std::is_same_v<kittens::fp8e5m2, T> ) || W%2 == 0 )
        #endif
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "shared_loadstore_gmem=fp8e5m2":
                                                      #endif
                                                                                         "shared_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {


        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void device_func(const GL &input, const GL &output,
                                                                                                      uint8_t *dpct_local) {
        auto __shm = (kittens::alignment_dummy *)
            dpct_local; // this is the CUDA shared memory
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            kittens::load <axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile,  input, {i, j, k, l});
            kittens::store<axis::value, false, ST, GL, kittens::coord<ST>>(output, shared_tile, {i, j, k, l});
        }
    }
};

template<typename T>
struct st_load_store_async {
    using dtype = T;
    template<int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64) 
        #ifdef KITTENS_HOPPER
        && ( ( !std::is_same_v<kittens::fp8e4m3, T> && !std::is_same_v<kittens::fp8e5m2, T> ) || W%2 == 0 )
        #endif
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_async_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_async_gmem=fp8e4m3":
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "shared_loadstore_async_gmem=fp8e5m2":
                                                      #endif
                                                                                         "shared_loadstore_async_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> static void device_func(const GL input, const GL output,
                                                                                                      uint8_t *dpct_local) {
        auto __shm = (kittens::alignment_dummy *)
            dpct_local; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        using ST = kittens::st<T, 16*H, 16*W>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            kittens::load_async<axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile, input, {i, j, k, l});
            kittens::load_async_wait();
            kittens::store<axis::value, false, ST, GL, kittens::coord<ST>>(output, shared_tile, {i, j, k, l});
        }
    }
};

using I0_t = std::integral_constant<int, 0>;
using I1_t = std::integral_constant<int, 1>;
using I2_t = std::integral_constant<int, 2>;
void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE, I0_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE, I0_t>::run(results);
}
#endif
