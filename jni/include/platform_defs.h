#pragma once

#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#elif defined(_MSC_VER)
    // MSVC doesn't have __builtin_expect; just pass through
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

#if defined(_MSC_VER)
    // Microsoft Visual C++ compiler
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    // GCC, Clang (Linux, macOS, etc.)
    #define RESTRICT __restrict__
#else
    // Fallback: just ignore the qualifier
    #define RESTRICT
#endif
