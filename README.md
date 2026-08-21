# wavesim

I wanted to understand what actually happens inside a GPU when a branch splits a wavefront, so I built a functional simulator of the SIMT execution model that AMD and NVIDIA GPUs use, and instrumented it with the counters that GPU programmers actually tune for: divergence, shared memory bank conflicts, and memory coalescing.

## The problem this models

A GPU does not run one thread at a time. It runs groups of 32 or 64 threads, called wavefronts, in lockstep on one program counter. When threads inside a wavefront disagree on a branch, the hardware cannot split them up. It runs both sides of the branch one after the other with an execution mask that switches lanes off. That mask dance is called divergence, and it is one of the three classic ways GPU code silently loses throughput. The other two live in the memory system: shared memory is split into banks that serialize when multiple lanes hit the same bank at different addresses, and global memory only reaches full bandwidth when the lanes of a wavefront touch contiguous cache lines that can be coalesced into a few wide transactions.

None of these effects are visible in ordinary CPU code, which is why I wanted a small machine where I could watch them happen with exact counters instead of profiler estimates.

## What I built

The core is a C++17 interpreter for a small custom register based IR that I made up for this project. It is deliberately not a real instruction set and has no binary encoding, it exists so kernels can be written in an assembly like style through a tiny C++ builder API. The machine model is:

* Wavefronts of configurable width, 32 by default, 64 supported, with a 64 bit exec mask.
* Per lane vector registers (VGPRs) and per wavefront scalar registers (SGPRs).
* Structured control flow through a divergence stack: IF, ELSE, ENDIF, LOOP, ENDLOOP. Compares write mask registers, IF intersects the mask with exec and pushes the saved state, ELSE inverts within the saved mask, ENDIF pops and restores.
* Workgroups made of several wavefronts with a barrier instruction. The scheduler runs each wave until it parks at a barrier or finishes, then releases the barrier once every unfinished wave has arrived. A barrier reached with a divergent mask is an error and the simulator throws.
* An LDS, local data share, with 32 word interleaved banks. Every 32 lane access phase counts the worst case number of distinct words mapped to one bank, and anything beyond 1 is recorded as serialized conflict cycles. Lanes reading the same address in a bank broadcast and do not conflict.
* A global memory model that collapses each 32 lane access phase into the set of unique 64 byte lines it touches and counts those as transactions. Contiguous aligned lanes coalesce into 2 transactions per phase, a 64 byte strided pattern degrades to 32.

Wave64 memory accesses are processed as two 32 lane phases, which mirrors how wide waves issue in halves on real hardware and keeps the counters comparable between widths.

An important caveat up front: the LDS and coalescing models are statistics counters bolted onto a functional interpreter. They tell you how much serialization or memory traffic a pattern would cause, they do not simulate time. There are no pipelines, caches, or latencies here, and none of the numbers are cycle accurate performance predictions.

## The kernels

Eleven kernels are written in the IR, in `kernels/kernels.cpp`, covering the classic patterns plus two deliberately bad ones so the counters can be shown moving in both directions:

vecadd (int and float), saxpy, a data dependent branch kernel (odd lanes do 3n+1, even lanes halve), a workgroup parallel reduction over LDS with barriers, a naive 64x64 matrix multiply, an LDS tiled 64x64 matrix multiply with 16x16 tiles, a coalesced copy, a strided copy that reads one word per cache line, and an LDS access pattern where every lane lands in bank 0.

Every kernel is verified bit exact against a scalar C++ oracle running the same arithmetic in the same per element order. For the float reduction the oracle replays the exact LDS tree order rather than a naive left to right sum, since float addition is not associative and a different order would legitimately differ in the last bit. The matmul oracles use fused multiply add in ascending k order to match the kernels exactly.

## What the counters show

Numbers below are from `make bench` on my machine, wave32, seed 42. Full data including the wave64 runs is committed in `results/results.json` and discussed in `RESULTS.md`.

| kernel | instructions | divergence | LDS conflict cycles | transactions per access |
|---|---|---|---|---|
| vecadd_f32 | 1152 | 0% | 0 | 2.00 |
| divergent_collatz_i32 | 1792 | 30.0% | 0 | 2.00 |
| reduce_f32 | 15360 | 13.5% | 0 | 1.89 |
| matmul_naive_f32 | 116864 | 0% | 0 | 1.50 |
| matmul_tiled_f32 | 77824 | 0% | 0 | 2.00 |
| copy_coalesced_f32 | 768 | 0% | 0 | 2.00 |
| copy_strided_f32 | 896 | 0% | 0 | 17.00 |
| lds_conflict_heavy_i32 | 1024 | 0% | 7936 | 2.00 |

A few things I found genuinely interesting in these:

The tiled matmul cuts total global transactions from 24832 to 2304, a 10.8x reduction, which is the whole point of tiling. But its transactions per access is actually worse than the naive kernel, 2.00 against 1.50. The naive kernel gets a flattering per access ratio because its A row loads broadcast the same address across half a wave. Tiling wins on total traffic, not on the per access ratio, and if I had only logged the ratio I would have concluded the naive kernel was better behaved. Counters need to be read carefully.

The strided copy shows 17.00 transactions per access rather than the theoretical worst case 32, because its coalesced store phases average against the fully scattered load phases. The load phases alone measure 32 lines each, which the hand computed cases in the results file confirm.

The reduction shows 13.5% divergence at wave32 but 27.4% at wave64, same algorithm, same data. The tail of the reduction tree keeps fewer active lanes than a wide wave has, so wider waves spend proportionally more issues partially masked. That tradeoff is a real consideration when picking wave width for control heavy code.

The bank conflict kernel costs 31 extra serialized cycles for every one of its 256 access phases, 7936 total, while the tiled matmul with its carefully laid out 16x16 tiles measures exactly zero.

## Verifying it

`make test` builds and runs 58 tests with about 1500 assertions: per instruction semantics against host arithmetic, nested divergence including if inside loop and exec restoration, cross wave communication through a barrier, error cases that must throw, hand computed LDS and coalescing patterns (stride 1, stride 2, stride 32, broadcast, partial masks, wave64 phases), all eleven kernels bit exact against their oracles, wave32 against wave64 output equality, and determinism, same seed twice must give identical outputs and identical stats.

```
make        # build tests and bench
make test   # run the test suite
make bench  # regenerate results/results.json
```

No dependencies beyond a C++17 compiler and make.

## Limitations

* This is a statistics model, not a timing model. It counts what would serialize or how many transactions a pattern generates. It does not predict runtime.
* The IR is my own invention for this project. It borrows the concepts and vocabulary of real GPU ISAs, exec masks, SGPRs, VGPRs, LDS, but no real instruction set is implemented or emulated.
* Instructions per second of the simulator itself, reported in the results, is single threaded interpreter throughput on my specific machine, an Apple Silicon laptop. It is a property of my interpreter, not of any modeled hardware, and will differ on other machines.
* One workgroup executes at a time and waves within it run round robin between barriers. That is enough for functional correctness, it is not a model of hardware scheduling.
* Global memory is a flat word addressed array with no caches. The coalescing counter models line granularity only.
