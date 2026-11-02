/*
 * mc_pi.c - Monte Carlo estimation of pi, map-reduce style with pthreads.
 *
 * Build:  gcc -O2 -pthread -o mc_pi mc_pi.c
 * Usage:  ./mc_pi <nPoints> <nThreads> [runs=5] [seed=12345]
 *
 * Output (one CSV line):
 *   nPoints,nThreads,runs,avg_time_sec,min_time_sec,max_time_sec,pi_estimate
 *
 * Design
 *   MAP    : thread t draws its own share of points (nPoints/N, remainder
 *            spread over the first threads) using a PRIVATE RNG state and
 *            counts hits in a PRIVATE local variable.
 *   REDUCE : after pthread_join, main sums the per-thread counters.
 *
 * No shared variable is touched inside the hot loop. Each thread's result
 * slot is padded to a full cache line to avoid false sharing when the
 * result is written at the end.
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CACHE_LINE 64

/* ---------- per-thread RNG: xoshiro256**, seeded via splitmix64 ---------- */
typedef struct { uint64_t s[4]; } rng_t;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&seed);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

/* uniform double in [0,1) using the top 53 bits */
static inline double rng_double(rng_t *r) {
    return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* ---------- thread data ---------- */
typedef struct {
    uint64_t nSamples;   /* input  : samples for this thread   */
    uint64_t seed;       /* input  : unique seed               */
    uint64_t hits;       /* output : local count inside circle */
    char pad[CACHE_LINE - 3 * sizeof(uint64_t)];
} __attribute__((aligned(CACHE_LINE))) task_t;

static void *worker(void *arg) {
    task_t *t = (task_t *)arg;
    rng_t rng;
    rng_seed(&rng, t->seed);          /* independent generator */

    uint64_t local = 0;               /* private counter */
    for (uint64_t i = 0; i < t->nSamples; i++) {
        double x = rng_double(&rng);
        double y = rng_double(&rng);
        if (x * x + y * y <= 1.0) local++;
    }
    t->hits = local;                  /* single write at the end */
    return NULL;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <nPoints> <nThreads> [runs=5] [seed=67]\n", argv[0]);
        return 1;
    }
    uint64_t nPoints  = strtoull(argv[1], NULL, 10);
    int      nThreads = atoi(argv[2]);
    int      runs     = (argc > 3) ? atoi(argv[3]) : 5;
    uint64_t baseSeed = (argc > 4) ? strtoull(argv[4], NULL, 10) : 12345ULL;
    if (nPoints == 0 || nThreads < 1 || runs < 1) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    /* ---- initialisation (NOT timed) ---- */
    pthread_t *tid = malloc(sizeof(pthread_t) * nThreads);
    task_t *tasks = aligned_alloc(CACHE_LINE, sizeof(task_t) * nThreads);
    if (!tid || !tasks) { perror("alloc"); return 1; }

    double sum = 0, tmin = 1e30, tmax = 0;
    double pi = 0;

    for (int r = 0; r < runs; r++) {
        uint64_t base = nPoints / nThreads, rem = nPoints % nThreads;
        for (int i = 0; i < nThreads; i++) {
            tasks[i].nSamples = base + ((uint64_t)i < rem ? 1 : 0);
            /* distinct seed per thread AND per run */
            tasks[i].seed = baseSeed + (uint64_t)r * 1000003ULL + (uint64_t)i * 7919ULL;
            tasks[i].hits = 0;
        }

        /* ---- timed region: create threads, compute, join, reduce ---- */
        double t0 = now_sec();
        for (int i = 0; i < nThreads; i++)
            if (pthread_create(&tid[i], NULL, worker, &tasks[i]) != 0) {
                perror("pthread_create"); return 1;
            }
        for (int i = 0; i < nThreads; i++) pthread_join(tid[i], NULL);

        uint64_t total = 0;                       /* REDUCE */
        for (int i = 0; i < nThreads; i++) total += tasks[i].hits;
        double t1 = now_sec();
        /* ------------------------------------------------------------- */

        double dt = t1 - t0;
        sum += dt;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
        pi = 4.0 * (double)total / (double)nPoints;
    }

    /* single CSV line: all I/O happens after timing */
    printf("%llu,%d,%d,%.6f,%.6f,%.6f,%.8f\n",
           (unsigned long long)nPoints, nThreads, runs,
           sum / runs, tmin, tmax, pi);

    free(tid);
    free(tasks);
    return 0;
}
