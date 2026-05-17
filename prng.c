#include "hash.h"
#include "prng.h"

_Thread_local uint64_t prng_state[4];

void prng_seed(uint64_t seed) {
    // expand one 64-bit seed into the 4-word xoshiro state via splitmix64
    prng_state[0] = splitmix64(seed);
    prng_state[1] = splitmix64(seed + 0x9E3779B97F4A7C15ULL);
    prng_state[2] = splitmix64(seed + 0x3C6EF372FE94F82AULL);
    prng_state[3] = splitmix64(seed + 0xDAA66D2C7DDF743FULL);

    // xoshiro requires non-zero state
    if (!(prng_state[0] | prng_state[1] | prng_state[2] | prng_state[3])) prng_state[0] = 1;
}
