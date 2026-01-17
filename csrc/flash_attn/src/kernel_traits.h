/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

// NOTE: Do NOT use "using namespace cute;" at global scope here!
// It causes ambiguity with at::Layout from PyTorch headers.

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = cute::Copy_Atom<cute::DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = cute::Copy_Atom<cute::DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = cute::TiledMMA<
        typename Base::MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<kNWarps>, cute::_1, cute::_1>>,  // 4x1x1 or 8x1x1 thread group
        cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_16>>;

    using SmemLayoutAtomQ = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutQ = decltype(cute::tile_to_shape(
        SmemLayoutAtomQ{},
        cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(cute::tile_to_shape(
        SmemLayoutAtomQ{},
        cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}));

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    using SmemLayoutVtransposed = decltype(
        cute::composition(SmemLayoutKV{}, cute::make_layout(cute::Shape<cute::Int<kHeadDim>, cute::Int<kBlockN>>{}, cute::GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(cute::detail::get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    cute::Layout<cute::Shape<cute::Int<8>, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtomO{},
        cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{}));
    using SmemCopyAtomO = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomOaccum = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kSmemQSize = cute::size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = cute::size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = cute::Layout<cute::Shape <cute::Int<kNThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
                                  cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        cute::AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyQKV = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));  // Val layout, 8 vals per read

    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::Int<kGmemRowsPerThread>, cute::_8>, cute::Stride<cute::_8, cute::_1>>{}));
    using GmemTiledCopyO = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        cute::Layout<cute::Shape <cute::_16, cute::_8>,  // Thread layout, 8 threads per row
               cute::Stride< cute::_8, cute::_1>>,
        cute::Layout<cute::Shape <cute::_8, cute::_16>,  // Thread layout, 16 threads per row
               cute::Stride< cute::_16, cute::_1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        cute::Layout<cute::Shape < cute::_1, cute::_4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        cute::Layout<cute::Shape < cute::_1, cute::_4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        cute::Layout<cute::Shape < cute::_1, cute::_8>>{}));  // Val layout, 8 vals per load
    using GmemTiledCopyRotcossinPaged = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        cute::Layout<cute::Shape<cute::Int<kGmemRowsPerThread>, cute::_4>, cute::Stride<cute::_4, cute::_1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        cute::Layout<cute::Shape<cute::Int<kGmemRowsPerThread>, cute::_8>, cute::Stride<cute::_8, cute::_1>>{}));  // Val layout, 8 vals per load
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = cute::TiledMMA<
        typename Base::MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<AtomLayoutMSdP>, cute::Int<kNWarps / AtomLayoutMSdP>, cute::_1>>,
        cute::Tile<cute::Int<16 * AtomLayoutMSdP>, cute::Int<16 * kNWarps / AtomLayoutMSdP>, cute::_16>>;

    using TiledMmadKV = cute::TiledMMA<
        typename Base::MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<AtomLayoutNdKV>, cute::Int<kNWarps / AtomLayoutNdKV>, cute::_1>>,
        cute::Tile<cute::Int<16 * AtomLayoutNdKV>, cute::Int<16 * kNWarps / AtomLayoutNdKV>, cute::_16>>;

    using TiledMmadQ = cute::TiledMMA<
        typename Base::MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<AtomLayoutMdQ>, cute::Int<kNWarps / AtomLayoutMdQ>, cute::_1>>,  // 2x4x1 or 4x2x1 thread group
        cute::Tile<cute::Int<16 * AtomLayoutMdQ>, cute::Int<16 * kNWarps / AtomLayoutMdQ>, cute::_16>>;

    using SmemLayoutAtomQdO = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutQdO = decltype(cute::tile_to_shape(
        SmemLayoutAtomQdO{},
        cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{})));

    using SmemLayoutAtomKV = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    cute::Layout<cute::Shape<cute::Int<kBlockM / kNWarps>, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutKV = decltype(cute::tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        cute::composition(SmemLayoutKV{}, cute::make_layout(cute::Shape<cute::Int<kHeadDim>, cute::Int<kBlockN>>{}, cute::GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(cute::detail::get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    // static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        cute::composition(cute::Swizzle<kSwizzlePdS, 3, 3>{},
                    cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<kPBlockN>>,
                           cute::Stride<cute::Int<kPBlockN>, cute::_1>>{}));
    using SmemLayoutPdS = decltype(cute::tile_to_shape(
        SmemLayoutAtomPdS{},
        cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        cute::composition(SmemLayoutPdS{}, cute::make_layout(cute::Shape<cute::Int<kBlockN>, cute::Int<kBlockM>>{}, cute::GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(cute::detail::get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        cute::composition(SmemLayoutQdO{}, cute::make_layout(cute::Shape<cute::Int<kHeadDim>, cute::Int<kBlockM>>{}, cute::GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(cute::detail::get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdKV = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutdKV = decltype(cute::tile_to_shape(
        SmemLayoutAtomdKV{},
        cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kHeadDim>{})));
    using SmemCopyAtomdKV = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        cute::composition(cute::Swizzle<kSwizzle, 3, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                           cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
    using SmemLayoutdQ = decltype(cute::tile_to_shape(
        SmemLayoutAtomdQ{},
        cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kHeadDim>{})));
    using SmemCopyAtomdQ = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = cute::size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = cute::size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = cute::size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize = cute::size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = cute::size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + kSmemPSize
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = cute::Layout<cute::Shape <cute::Int<kNThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
                                  cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        cute::AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyQKV = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape < cute::_1, cute::_8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydKV = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape < cute::_1, cute::_8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape < cute::_1, cute::_8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        cute::Layout<cute::Shape <cute::_32, cute::_8>,  // Thread layout, 8 threads per row
               cute::Stride< cute::_8, cute::_1>>,
        cute::Layout<cute::Shape <cute::_16, cute::_16>,  // Thread layout, 16 threads per row
               cute::Stride< cute::_16, cute::_1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        cute::Layout<cute::Shape < cute::_1, cute::_4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        cute::Layout<cute::Shape <cute::_8, cute::_32>,  // Thread layout, 8 threads per row
                               cute::Stride<cute::_32, cute::_1>>{},
                        cute::Layout<cute::Shape < cute::_1, cute::_1>>{}));  // Val layout, 1 val per store

};

////////////////////////////////////////////////////////////////////////////////////////////////////