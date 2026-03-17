#include "psgn.h"

uint8_t *sdr_create(int size) { return (uint8_t *)calloc(size, 1); }
void sdr_free(uint8_t *s) { free(s); }
void sdr_copy(uint8_t *dst, const uint8_t *src, int size) { memcpy(dst, src, size); }
void sdr_clear(uint8_t *s, int size) { memset(s, 0, size); }

void sdr_or(uint8_t *dst, const uint8_t *a, const uint8_t *b, int size) {
    for (int i = 0; i < size; i++) dst[i] = a[i] | b[i];
}
void sdr_or_inplace(uint8_t *dst, const uint8_t *src, int size) {
    for (int i = 0; i < size; i++) dst[i] |= src[i];
}
void sdr_and(uint8_t *dst, const uint8_t *a, const uint8_t *b, int size) {
    for (int i = 0; i < size; i++) dst[i] = a[i] & b[i];
}
void sdr_xor_inplace(uint8_t *dst, const uint8_t *src, int size) {
    for (int i = 0; i < size; i++) dst[i] ^= src[i];
}
void sdr_xor_shifted_inplace(uint8_t *dst, const uint8_t *src, int size, int shift) {
    int mask = size - 1; /* Assumes size is power of 2 */
    int s = shift & mask;
    for (int i = 0; i < size; i++)
        dst[i] ^= src[(i - s + size) & mask];
}
void sdr_or_shifted_inplace(uint8_t *dst, const uint8_t *src, int size, int shift) {
    int mask = size - 1;
    int s = shift & mask;
    for (int i = 0; i < size; i++)
        dst[i] |= src[(i - s + size) & mask];
}

int sdr_count(const uint8_t *s, int size) {
    int c = 0; 
    for (int i = 0; i < size; i++) c += __builtin_popcount(s[i]); 
    return c;
}

void sdr_circular_shift(uint8_t *dst, const uint8_t *src, int size, int shift) {
    int mask = size - 1;
    int s = shift & mask;
    for (int i = 0; i < size; i++)
        dst[i] = src[(i - s + size) & mask];
}

/* High-entropy random sampling for large VSA spaces (>32k) */
void sdr_random_sample(int n, int k, int *out) {
    int *pool = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) pool[i] = i;
    for (int i = 0; i < k; i++) {
        /* Use arc4random for full 32-bit entropy to avoid bit-clustering in large SDRs */
        uint32_t r = arc4random();
        int j = i + (r % (n - i));
        int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        out[i] = pool[i];
    }
    free(pool);
}
