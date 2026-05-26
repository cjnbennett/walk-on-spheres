// implements prng using xoshiro256++
#pragma once
#include <stdint.h>

namespace wos {

extern uint64_t prng_state[4];

void prng_seed(uint64_t seed);

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// raw 64-bit uniform random integer
static inline uint64_t prng_u64(void) {
    const uint64_t result = rotl64(prng_state[0] + prng_state[3], 23) + prng_state[0];
    const uint64_t t = prng_state[1] << 17;
    prng_state[2] ^= prng_state[0];
    prng_state[3] ^= prng_state[1];
    prng_state[1] ^= prng_state[2];
    prng_state[0] ^= prng_state[3];
    prng_state[2] ^= t;
    prng_state[3] = rotl64(prng_state[3], 45);
    return result;
}

// uniform double in [0, 1) - use top 53 bits (double mantissa precision)
static inline double prng_unit(void) {
    return (prng_u64() >> 11) * (1.0 / (1ULL << 53));
}

}
