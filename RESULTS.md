# Results notes

Personal log of measured numbers so I do not have to rerun everything to quote them. All numbers below came from actual runs on my machine. Machine and toolchain: Apple Silicon (arm64) laptop, single threaded runs, Apple clang 21.0.0 with -O2. Simulator statistics (instructions, divergence, conflicts, transactions) are machine independent and deterministic for a given seed. The instr/sec column is interpreter throughput of the simulator process itself on this machine, it is not a modeled GPU speed and it will vary elsewhere.

Reproduce:

```
make test    # 58 tests, 1481 assertions, all passing as of this log
make bench   # rewrites results/results.json, prints the summary table
```

Bench defaults: seed 42, kernels run at both wave32 and wave64. Timing loop targets about 0.5 s per kernel after a warmup launch, instr/sec = instructions per launch times reps divided by measured wall time of the launches only (input setup and oracle checking excluded).

## Test suite

* `make test`: 58 tests, 1481 checks, 0 failures.
* All 11 kernels bit exact against their scalar oracles at wave32 and the 4 equivalence kernels also at wave64 (the bench additionally verifies oracle match for all 11 at wave64, all true in results.json).
* Determinism: saxpy and tiled matmul run twice with seed 7, identical output hashes and identical stats structs. Different seed changes the hash. Bench re confirms with tiled matmul.

## Per kernel, wave32, seed 42

| kernel | instr | vector instr | div % | lds acc | lds conflict cyc | glb acc | glb trans | trans/acc | instr/sec |
|---|---|---|---|---|---|---|---|---|---|
| vecadd_i32 | 1152 | 1024 | 0.00 | 0 | 0 | 384 | 768 | 2.00 | 15.7M |
| vecadd_f32 | 1152 | 1024 | 0.00 | 0 | 0 | 384 | 768 | 2.00 | 17.8M |
| saxpy_f32 | 1024 | 896 | 0.00 | 0 | 0 | 384 | 768 | 2.00 | 17.1M |
| divergent_collatz_i32 | 1792 | 1280 | 30.00 | 0 | 0 | 256 | 512 | 2.00 | 21.1M |
| reduce_i32 | 15360 | 4032 | 13.49 | 720 | 0 | 144 | 272 | 1.89 | 63.8M |
| reduce_f32 | 15360 | 4032 | 13.49 | 720 | 0 | 144 | 272 | 1.89 | 63.6M |
| matmul_naive_f32 | 116864 | 91392 | 0.00 | 0 | 0 | 16512 | 24832 | 1.50 | 25.4M |
| matmul_tiled_f32 | 77824 | 74368 | 0.00 | 17408 | 0 | 1152 | 2304 | 2.00 | 20.4M |
| copy_coalesced_f32 | 768 | 640 | 0.00 | 0 | 0 | 256 | 512 | 2.00 | 14.2M |
| copy_strided_f32 | 896 | 768 | 0.00 | 0 | 0 | 256 | 4352 | 17.00 | 12.9M |
| lds_conflict_heavy_i32 | 1024 | 768 | 0.00 | 256 | 7936 | 128 | 256 | 2.00 | 9.1M |

Problem sizes: 1d kernels 4096 elements, 16 workgroups of 256 threads. Matmuls 64x64 times 64x64, 16 workgroups of 256 threads, 16x16 LDS tiles. Reduction output is 16 partial sums, one per workgroup.

## Per kernel, wave64, seed 42 (differences that matter)

* Instruction issues drop to roughly half everywhere since one wave64 issue covers 64 lanes, for example vecadd 1152 to 576, tiled matmul 77824 to 38912, naive matmul 116864 to 58432.
* Outputs identical to wave32 for all 11 kernels (fnv1a hashes equal, listed under wave32_vs_wave64_outputs_identical, all true).
* Memory phase counters unchanged (a wave64 access counts as two 32 lane phases), so lds and global numbers match wave32 exactly, including 7936 conflict cycles for the conflict heavy kernel.
* Divergence changes for real: reduce goes 13.49% at wave32 to 27.40% at wave64, collatz stays 30.00%. The reduction tree tail keeps few lanes active, so wider waves are partially masked more often.
* Interpreter instr/sec is lower at wave64 (each issue does 64 lanes of work), for example reduce 63.8M to 34.1M.

## Matmul tiling comparison, wave32

* Global accesses: naive 16512, tiled 1152 (14.3x fewer).
* Global transactions: naive 24832, tiled 2304 (10.8x fewer).
* Transactions per access: naive 1.50, tiled 2.00. Naive looks better on the ratio only because its A row loads broadcast one address across half a wave. Total traffic is the honest metric here, the ratio alone is misleading.
* LDS conflict cycles: tiled 0 across 17408 access phases. The 16x16 tile layout is conflict free in this banking model.
* Instructions: naive 116864, tiled 77824 (per wave loop overhead amortized over the tile).

## Hand computed cases (all measured equal to expectation, also asserted in tests)

* LDS, wave32: stride 1 word gives 0 conflict cycles, stride 2 gives 1, stride 32 gives 31 (full serialization into bank 0), broadcast of one address gives 0, stride 32 with only 8 active lanes gives 7.
* LDS, wave64: stride 1 gives 0 across 2 phases, stride 32 gives 62 (31 per phase).
* Global, wave32: contiguous 4 byte lanes give 2 transactions (128 aligned bytes over 64 byte lines), 8 byte stride gives 4, 64 byte stride gives 32 (worst case), broadcast gives 1, contiguous with 8 active lanes gives 1.
* Global, wave64: contiguous gives 4 transactions over 2 phases.

## Aggregate

Total simulated instruction issues across the 11 wave32 bench runs: 233,216 per full pass (116,608 at wave64), checked against results.json by summing the instructions field. Interpreter throughput ranged 9.1M to 63.8M instr/sec at wave32 on this machine depending on the mix of memory instructions (LDS conflict scanning is the slowest path) versus ALU.

## Caveats I want to remember

* Stats model, not timing. Nothing here predicts cycles or seconds on real hardware.
* Custom IR, not a real ISA. Concepts match public GPU architecture material, encodings and semantics are my own.
* instr/sec is single thread performance of this interpreter on this specific laptop.
* saxpy updates y in place, so repeated bench launches change values but not access patterns or stats. Oracle checking always uses a fresh first launch.
