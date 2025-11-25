#pragma once

// For TORCH_CHECK
#include <c10/util/Exception.h>

// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_ONLY_HEAD_DIM
  #define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    TORCH_CHECK(HEADDIM == FLASHATTENTION_ONLY_HEAD_DIM, "FlashAttention compiled with fixed head dim ", FLASHATTENTION_ONLY_HEAD_DIM, " but received ", HEADDIM); \
    constexpr static int kHeadDim = FLASHATTENTION_ONLY_HEAD_DIM; \
    return __VA_ARGS__();                \
  }()
#else
  #define HEADDIM_SWITCH(HEADDIM, ...)   \
    [&] {                                    \
      if (HEADDIM <= 32) {                   \
        constexpr static int kHeadDim = 32;  \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 64) {            \
        constexpr static int kHeadDim = 64;  \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 96) {            \
        constexpr static int kHeadDim = 96;  \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 128) {           \
        constexpr static int kHeadDim = 128; \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 160) {           \
        constexpr static int kHeadDim = 160; \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 192) {           \
        constexpr static int kHeadDim = 192; \
        return __VA_ARGS__();                \
      } else if (HEADDIM <= 256) {           \
        constexpr static int kHeadDim = 256; \
        return __VA_ARGS__();                \
      }                                      \
    }()
  #endif

#ifdef FLASHATTENTION_ONLY_BF16
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
        TORCH_CHECK(!COND, "FlashAttention compiled with only one precision but received both."); \
        using elem_type = cutlass::bfloat16_t;     \
        return __VA_ARGS__();                  \
    }()
#else
  #define FP16_SWITCH(COND, ...)               \
    [&] {                                      \
      if (COND) {                              \
        using elem_type = cutlass::half_t;     \
        return __VA_ARGS__();                  \
      } else {                                 \
        using elem_type = cutlass::bfloat16_t; \
        return __VA_ARGS__();                  \
      }                                        \
    }()
#endif