#include "registration.h"
#include "pytorch_shim.h"
#include "namespace_config.h"

#include <c10/core/DeviceType.h>
#include <torch/library.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
/**
 *  Externs for the flash_attn ops to be exposed as a pytorch library
 */

namespace FLASH_NAMESPACE {

////////////////////////////// From flash_api.cpp //////////////////////////////

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);

std::vector<at::Tensor>
flexi_direct_mha_varlen_fwd(at::Tensor &q,
               const at::Tensor &k_meta,
               const at::Tensor &v_meta,
               const int64_t num_blocks,
               const at::Tensor &k_ptr_table,
               const at::Tensor &v_ptr_table,
               std::optional<at::Tensor> &out_,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               std::optional<at::Tensor> &seqused_k,
               std::optional<const at::Tensor> &leftpad_k_,
               std::optional<at::Tensor> &alibi_slopes_,
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);


std::vector<at::Tensor>
flexi_mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k_meta,  // representative tensor for shape/stride
               const at::Tensor &v_meta,  // representative tensor for shape/stride
               const int64_t num_blocks,  // number of pages
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_,
               int64_t cached_k_ptrs=0,
               int64_t cached_v_ptrs=0);


/////////////////////////// From flash_api_sparse.cpp //////////////////////////
std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits);

std::tuple<int64_t, int64_t> prepare_flexi_kv_ptrs(
    const std::vector<at::Tensor>& k_list,
    const std::vector<at::Tensor>& v_list);

void free_flexi_kv_ptrs(int64_t k_ptrs_dev, int64_t v_ptrs_dev);

#ifndef FLASHATTENTION_DISABLE_SPARSE
std::vector<at::Tensor>
mha_fwd_sparse(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &block_count,
               const at::Tensor &block_offset,
               const at::Tensor &column_count,
               const at::Tensor &column_index,
               const std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
               const std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
               const double p_dropout,
               const double softmax_scale,
               bool is_causal,
               const double softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                      const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &block_count,
                      const at::Tensor &block_offset,
                      const at::Tensor &column_count,
                      const at::Tensor &column_index,
                      const std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                      const at::Tensor &cu_seqlens_q,  // b+1
                      const at::Tensor &cu_seqlens_k,  // b+1
                      const std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                      const std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                      int64_t max_seqlen_q,
                      const int64_t max_seqlen_k,
                      const double p_dropout,
                      const double softmax_scale,
                      const bool zero_tensors,
                      bool is_causal,
                      const double softcap,
                      const bool return_softmax,
                      std::optional<at::Generator> gen_);
#endif

/**
 *  Torch Library Registration
 */
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd));

    ops.def("flexi_varlen_fwd(Tensor! q, Tensor k_meta, Tensor v_meta, int num_blocks, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen, int! cached_k_ptrs, int! cached_v_ptrs) -> Tensor[]");
    ops.impl("flexi_varlen_fwd", torch::kCUDA, make_pytorch_shim(&flexi_mha_varlen_fwd));

    ops.def("flexi_direct_varlen_fwd(Tensor! q, Tensor k_meta, Tensor v_meta, int num_blocks, "
            "Tensor k_ptr_table, Tensor v_ptr_table, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("flexi_direct_varlen_fwd", torch::kCUDA, make_pytorch_shim(&flexi_direct_mha_varlen_fwd));

    ops.def("fwd_kvcache(Tensor! q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, "
            "Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? leftpad_k, Tensor? block_table, "
            "Tensor? alibi_slopes, Tensor!? out, float softmax_scale, bool is_causal, int window_size_left, "
            "int window_size_right, float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]");
    ops.impl("fwd_kvcache", torch::kCUDA, make_pytorch_shim(&mha_fwd_kvcache));

    ops.def("prepare_flexi_kv_ptrs(Tensor[] k_list, Tensor[] v_list) -> (int!, int!)");
    ops.impl("prepare_flexi_kv_ptrs", torch::kCUDA, &prepare_flexi_kv_ptrs);
    // No return value; schema still needs the arrow. Implementation registered in a CatchAll block below.
    ops.def("free_flexi_kv_ptrs(int! k_ptrs_dev, int! v_ptrs_dev) -> ()");



#ifndef FLASHATTENTION_DISABLE_SPARSE
    ops.def("fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor? alibi_slopes, "
            "float p_dropout, float softmax_scale, bool is_causal, "
            "float softcap, bool return_softmax, Generator? gen)"
            "-> Tensor[]");
    ops.impl("fwd_sparse", torch::kCUDA, &mha_fwd_sparse);

    ops.def("varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd_sparse", torch::kCUDA, &mha_varlen_fwd_sparse);
#endif

}

// Catch-all implementation for the pointer free routine (no tensor inputs).
TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CatchAll, m) {
    m.impl("free_flexi_kv_ptrs", &free_flexi_kv_ptrs);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
}
