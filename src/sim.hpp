#pragma once

// Wavefront execution model with LDS bank conflict and global memory
// coalescing statistics. These are statistics counters layered over a
// functional interpreter, not a timing or cycle accurate model.

#include "ir.hpp"

#include <cstdint>
#include <vector>

namespace wavesim {

constexpr int kNumVgpr = 32;
constexpr int kNumSgpr = 32;
constexpr int kNumMreg = 8;
constexpr int kLdsBanks = 32;
constexpr uint32_t kCacheLineBytes = 64;

struct Stats {
  uint64_t total_instructions = 0;   // every issued instruction
  uint64_t vector_instructions = 0;  // vector ALU, cmp and memory issues with a
                                     // nonzero exec mask
  uint64_t divergent_vector_instructions = 0; // vector issues with a partial mask
  uint64_t scalar_instructions = 0;
  uint64_t control_instructions = 0;
  uint64_t lds_accesses = 0;          // 32 lane phases with any active lane
  uint64_t lds_conflict_cycles = 0;   // extra serialized cycles beyond 1 per phase
  uint64_t global_accesses = 0;       // 32 lane phases with any active lane
  uint64_t global_transactions = 0;   // 64 byte line transactions
  uint64_t barriers = 0;              // barrier instructions issued per wave

  void add(const Stats &o);
  bool operator==(const Stats &o) const;
};

class GlobalMemory {
public:
  explicit GlobalMemory(uint32_t bytes);

  // Bump allocator, returns a 64 byte aligned byte address.
  uint32_t alloc(uint32_t bytes);

  uint32_t loadWord(uint32_t byteAddr) const;
  void storeWord(uint32_t byteAddr, uint32_t value);

  uint32_t sizeBytes() const { return static_cast<uint32_t>(words_.size()) * 4; }

private:
  std::vector<uint32_t> words_;
  uint32_t bump_ = 0;
};

struct LaunchConfig {
  int wave_width = 32;       // 32 or 64
  int workgroup_size = 256;  // threads per workgroup, multiple of wave_width
  int num_workgroups = 1;
  uint32_t lds_bytes = 32 * 1024;
};

// Runs the program over the whole grid. Kernel arguments are copied into
// SGPRs starting at KernelBuilder::kSArg0 for every wavefront.
Stats runProgram(const Program &prog, const LaunchConfig &cfg, GlobalMemory &mem,
                 const std::vector<uint32_t> &args);

} // namespace wavesim
