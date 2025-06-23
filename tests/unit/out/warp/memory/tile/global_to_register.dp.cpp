#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "global_to_register.dp.hpp"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static void g2r_global_wrapper_2d(const GL &input, const GL &output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct g2r_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int SIZE = H*W*256 * B * D * R * C;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, D, -1, 16*C*W>;
            GL input (d_i, B, nullptr, 16*R*H, nullptr);
            GL output(d_o, B, nullptr, 16*R*H, nullptr);
            // run kernel
            /*
            DPCT1026:79: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
            /*
            DPCT1049:24: The work-group size passed to the SYCL kernel may
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
                                                          NUM_WORKERS, GL,
                                                          args...>(input,
                                                                   output);
                                  });
                        });
                  }
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2r_sweep_size_2d = loop_h<g2r_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using g2r_sweep_size_2d_warp = g2r_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;

template<typename T>
struct load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "reg_loadstore_gmem=half" :
                                                    //   std::is_same_v<T, kittens::fp8e4m3> ? "reg_loadstore_gmem=fp8e4m3" :
                                                                                         "reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    /*
    DPCT1110:25: The total declared local variable size in device function
    device_func exceeds 128 bytes and may cause high register pressure. Consult
    with your hardware vendor to find the total register size available and
    adjust the code, or use smaller sub-group size to avoid high register
    pressure.
    */
    template <int H, int W, int NW, kittens::ducks::gl::all GL,
              kittens::ducks::rt_layout::all L>
    static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        for(int i = 0; i < input.batch(); i++) for(int j = 0; j < input.depth(); j++) for(int k = 0; k < input.rows()/reg_tile.rows; k++) for(int l = 0; l < input.cols()/reg_tile.cols; l++) {
            kittens::load(reg_tile, input, {i, j, k, l});
            kittens::store(output, reg_tile, {i, j, k, l});
        }
    }
};

void warp::memory::tile::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    g2r_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::bf16>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::bf16>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::half>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    g2r_sweep_size_2d_warp<load_store<kittens::half>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // g2r_sweep_size_2d_warp<load_store<kittens::fp8e4m3>, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // g2r_sweep_size_2d_warp<load_store<kittens::fp8e4m3>, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
}

#endif