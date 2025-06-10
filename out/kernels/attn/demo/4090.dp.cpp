#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 1024; 
constexpr int ATTN_D = 64;
constexpr int ITER   = 10;

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3; 

template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

template <int D>
SYCL_EXTERNAL void attend_ker(const globals<D> g, uint8_t *dpct_local) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    using load_group =
        kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = item_ct1.get_group(0), head = item_ct1.get_group(1),
              q_seq = item_ct1.get_group(2) * NUM_WORKERS + workerid;

    auto __shm = (alignment_dummy *)dpct_local;
    shared_allocator al((int*)&__shm[0]);
    
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq*ROWS<D> < g.Qg.depth()) {
        load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});  // going through shared memory improves coalescing of dram reads.
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        load(q_reg, qo_smem[workerid]);
    }
    /*
    DPCT1065:309: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if constexpr (D == 64) q_reg *=
        sycl::ext::oneapi::bfloat16(0.125f * 1.44269504089f);
    else if constexpr (D == 128) q_reg *=
        sycl::ext::oneapi::bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});
    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        }
        else load_async_wait();
        /*
        DPCT1118:216: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:311: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            att_block = 0.f; // zero 16x16 attention tile
            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
            int first_index = (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D>; // one past the last KV index of this tile
            int start_fill = g.Kg.depth()-first_index < ROWS<D> ? g.Kg.depth()-first_index : ROWS<D>;
            right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());
            max_vec_last = max_vec;
            max_vec = max<axis::COL>(att_block, max_vec); 
            att_block = exp2(att_block - max_vec); 
            max_vec_last = exp2(max_vec_last - max_vec); 
            norm_vec *= max_vec_last; 
            norm_vec = sum<axis::COL>(att_block, norm_vec); 
            att_block_mma = att_block; // copy to bf16 tile
            load(v_reg, v_smem[subtile][tic]); 
            o_reg *= max_vec_last; 
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    /*
    DPCT1065:310: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (q_seq*ROWS<D> < g.Og.depth()) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}

#include "4090_harness.impl"