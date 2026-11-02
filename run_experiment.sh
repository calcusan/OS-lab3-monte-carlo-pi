#!/usr/bin/env bash
# run_experiments.sh - sweep thread counts, record time, compute speedup.
# Usage: ./run_experiments.sh [nPoints=400000000] [runs=5]
set -e

NPOINTS=${1:-100000000}
RUNS=${2:-5}
THREADS=(1 2 4 8 16 32 64 100 128)   # edit freely; keep 1 first (baseline)
OUT=results.csv

gcc -O2 -pthread -o mc_pi mc_pi.c

echo "# nPoints=$NPOINTS runs=$RUNS cores=$(nproc) date=$(date -Is)" > "$OUT"
echo "threads,avg_time,min_time,max_time,pi_estimate,speedup,efficiency" >> "$OUT"

T1=""
for N in "${THREADS[@]}"; do
    LINE=$(./mc_pi "$NPOINTS" "$N" "$RUNS")
    IFS=',' read -r _ th _ avg tmin tmax pi <<< "$LINE"
    [ -z "$T1" ] && T1=$avg
    SP=$(awk -v a="$T1" -v b="$avg" 'BEGIN{printf "%.3f", a/b}')
    EF=$(awk -v s="$SP" -v n="$th" 'BEGIN{printf "%.3f", s/n}')
    echo "$th,$avg,$tmin,$tmax,$pi,$SP,$EF" >> "$OUT"
    printf "threads=%-4s avg=%ss speedup=%s eff=%s pi=%s\n" "$th" "$avg" "$SP" "$EF" "$pi"
done

echo "Saved to $OUT"
