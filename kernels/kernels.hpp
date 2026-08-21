#pragma once

// Kernels written in the custom IR through the builder API, together with
// host side launch wrappers and scalar CPU oracles used for verification.

#include "../src/sim.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace wavesim {

// A kernel with its inputs staged in memory and its oracle output computed.
// Launching the program repeatedly on the same memory is safe for timing:
// every kernel writes the same addresses with the same values on a rerun,
// except saxpy which updates y in place but keeps identical access patterns.
struct PreparedKernel {
  std::string name;
  Program prog;
  LaunchConfig cfg;
  GlobalMemory mem{1024};
  std::vector<uint32_t> args;
  uint32_t out_base = 0;
  uint32_t out_len = 0; // in words
  std::vector<uint32_t> expected;
};

struct KernelRunResult {
  std::string name;
  Stats stats;
  bool oracle_match = false; // bit exact comparison against the scalar oracle
  uint64_t output_hash = 0;  // FNV1a over the output words
  uint32_t elements = 0;     // output size in words
};

// Kernel names:
//   vecadd_i32, vecadd_f32, saxpy_f32, divergent_collatz_i32,
//   reduce_i32, reduce_f32, matmul_naive_f32, matmul_tiled_f32,
//   copy_coalesced_f32, copy_strided_f32, lds_conflict_heavy_i32
const std::vector<std::string> &kernelNames();

PreparedKernel prepareKernel(const std::string &name, int wave_width,
                             uint32_t seed);

// Prepare, launch once, verify against the oracle.
KernelRunResult runKernel(const std::string &name, int wave_width,
                          uint32_t seed);

uint64_t hashWords(const std::vector<uint32_t> &words);

} // namespace wavesim
