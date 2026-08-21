#include "kernels.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace wavesim {

namespace {

using KB = KernelBuilder;

constexpr uint32_t kN = 4096;   // element count for 1d kernels
constexpr int kGroupSize = 256; // threads per workgroup
constexpr int kMatN = 64;       // matrix dimension
constexpr int kTile = 16;       // tile edge for the tiled matmul
constexpr uint32_t kMemBytes = 8u << 20;
constexpr uint32_t kLdsBytes = 32u * 1024;

std::vector<uint32_t> randomInts(std::mt19937 &rng, uint32_t n, uint32_t lo,
                                 uint32_t hi) {
  std::vector<uint32_t> out(n);
  std::uniform_int_distribution<uint32_t> dist(lo, hi);
  for (auto &v : out)
    v = dist(rng);
  return out;
}

std::vector<uint32_t> randomFloats(std::mt19937 &rng, uint32_t n) {
  std::vector<uint32_t> out(n);
  std::uniform_int_distribution<uint32_t> dist(0, 1u << 20);
  for (auto &v : out) {
    float f = static_cast<float>(dist(rng)) / 524288.0f - 1.0f; // [-1, 1]
    v = f32Bits(f);
  }
  return out;
}

void writeBuf(GlobalMemory &mem, uint32_t base, const std::vector<uint32_t> &v) {
  for (uint32_t i = 0; i < v.size(); i++)
    mem.storeWord(base + i * 4, v[i]);
}

LaunchConfig cfg1d(int wave_width) {
  LaunchConfig c;
  c.wave_width = wave_width;
  c.workgroup_size = kGroupSize;
  c.num_workgroups = static_cast<int>(kN) / kGroupSize;
  c.lds_bytes = kLdsBytes;
  return c;
}

// ---------------------------------------------------------------- builders

Program buildVecAdd(bool isFloat) {
  KB b(isFloat ? "vecadd_f32" : "vecadd_i32");
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4));
  b.vadd_i32(6, V(3), S(KB::kSArg0 + 1));
  b.global_load(7, V(6));
  if (isFloat)
    b.vadd_f32(8, V(5), V(7));
  else
    b.vadd_i32(8, V(5), V(7));
  b.vadd_i32(9, V(3), S(KB::kSArg0 + 2));
  b.global_store(V(9), V(8));
  b.end();
  return b.finish();
}

Program buildSaxpyF32() {
  KB b("saxpy_f32");
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4)); // x
  b.vadd_i32(6, V(3), S(KB::kSArg0 + 1));
  b.global_load(7, V(6)); // y
  b.vfma_f32(8, S(KB::kSArg0 + 2), V(5), V(7));
  b.global_store(V(6), V(8));
  b.end();
  return b.finish();
}

Program buildDivergentCollatzI32() {
  KB b("divergent_collatz_i32");
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4));
  b.vand_b32(6, V(5), Imm(1));
  b.vcmp_eq_i32(0, V(6), Imm(1));
  b.iff(0);
  b.vmul_i32(7, V(5), Imm(3));
  b.vadd_i32(7, V(7), Imm(1));
  b.els();
  b.vshr_b32(7, V(5), Imm(1));
  b.endif();
  b.vadd_i32(8, V(3), S(KB::kSArg0 + 1));
  b.global_store(V(8), V(7));
  b.end();
  return b.finish();
}

Program buildReduce(bool isFloat) {
  KB b(isFloat ? "reduce_f32" : "reduce_i32");
  const int vTid = KB::kVTidInGroup;
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4));
  b.vshl_b32(6, V(vTid), Imm(2)); // lds byte address for this thread
  b.lds_store(V(6), V(5));
  b.barrier();
  b.sshr_b32(20, S(KB::kSWorkgroupSize), Imm(1)); // stride
  b.loop();
  b.vcmp_lt_i32(0, V(vTid), S(20));
  b.iff(0);
  b.lds_load(7, V(6));
  b.vadd_i32(8, V(vTid), S(20));
  b.vshl_b32(9, V(8), Imm(2));
  b.lds_load(10, V(9));
  if (isFloat)
    b.vadd_f32(7, V(7), V(10));
  else
    b.vadd_i32(7, V(7), V(10));
  b.lds_store(V(6), V(7));
  b.endif();
  b.barrier();
  b.sshr_b32(20, S(20), Imm(1));
  b.vcmp_gt_i32(1, S(20), Imm(0));
  b.endloop(1);
  b.vcmp_eq_i32(2, V(vTid), Imm(0));
  b.iff(2);
  b.lds_load(11, Imm(0));
  b.vshl_b32(12, S(KB::kSWorkgroupId), Imm(2));
  b.vadd_i32(13, V(12), S(KB::kSArg0 + 1));
  b.global_store(V(13), V(11));
  b.endif();
  b.end();
  return b.finish();
}

Program buildMatmulNaiveF32() {
  KB b("matmul_naive_f32");
  const int vTid = KB::kVTidInGroup;
  b.sand_b32(20, S(KB::kSWorkgroupId), Imm(3)); // wgx
  b.sshr_b32(21, S(KB::kSWorkgroupId), Imm(2)); // wgy
  b.sshl_b32(22, S(21), Imm(4));                // wgy * 16
  b.sshl_b32(23, S(20), Imm(4));                // wgx * 16
  b.vand_b32(3, V(vTid), Imm(15));              // tx
  b.vshr_b32(4, V(vTid), Imm(4));               // ty
  b.vadd_i32(5, V(4), S(22));                   // row
  b.vadd_i32(6, V(3), S(23));                   // col
  b.vmov(7, FImm(0.0f));                        // acc
  b.smov(24, Imm(0));                           // k
  b.loop();
  b.vshl_b32(8, V(5), Imm(6)); // row * 64
  b.vadd_i32(8, V(8), S(24));
  b.vshl_b32(8, V(8), Imm(2));
  b.vadd_i32(8, V(8), S(KB::kSArg0 + 0));
  b.global_load(9, V(8)); // A[row][k]
  b.sshl_b32(25, S(24), Imm(6));
  b.vadd_i32(10, V(6), S(25));
  b.vshl_b32(10, V(10), Imm(2));
  b.vadd_i32(10, V(10), S(KB::kSArg0 + 1));
  b.global_load(11, V(10)); // B[k][col]
  b.vfma_f32(7, V(9), V(11), V(7));
  b.sadd_i32(24, S(24), Imm(1));
  b.vcmp_lt_i32(0, S(24), Imm(kMatN));
  b.endloop(0);
  b.vshl_b32(12, V(5), Imm(6));
  b.vadd_i32(12, V(12), V(6));
  b.vshl_b32(12, V(12), Imm(2));
  b.vadd_i32(12, V(12), S(KB::kSArg0 + 2));
  b.global_store(V(12), V(7));
  b.end();
  return b.finish();
}

Program buildMatmulTiledF32() {
  KB b("matmul_tiled_f32");
  const int vTid = KB::kVTidInGroup;
  const uint32_t btileBase = kTile * kTile * 4; // Btile after Atile in LDS
  b.sand_b32(20, S(KB::kSWorkgroupId), Imm(3)); // wgx
  b.sshr_b32(21, S(KB::kSWorkgroupId), Imm(2)); // wgy
  b.sshl_b32(22, S(21), Imm(4));                // wgy * 16
  b.sshl_b32(23, S(20), Imm(4));                // wgx * 16
  b.vand_b32(3, V(vTid), Imm(15));              // tx
  b.vshr_b32(4, V(vTid), Imm(4));               // ty
  b.vadd_i32(5, V(4), S(22));                   // row
  b.vadd_i32(6, V(3), S(23));                   // col
  b.vshl_b32(13, V(5), Imm(6));                 // row * 64
  b.vshl_b32(14, V(4), Imm(4));                 // ty * 16
  b.vshl_b32(15, V(vTid), Imm(2));              // Atile slot byte address
  b.vadd_i32(16, V(15), Imm(btileBase));        // Btile slot byte address
  b.vmov(7, FImm(0.0f));                        // acc
  b.smov(24, Imm(0));                           // t, tile index
  b.loop();
  b.sshl_b32(25, S(24), Imm(4)); // t * 16
  // Atile[ty][tx] = A[row][t*16 + tx]
  b.vadd_i32(8, V(3), S(25));
  b.vadd_i32(8, V(8), V(13));
  b.vshl_b32(8, V(8), Imm(2));
  b.vadd_i32(8, V(8), S(KB::kSArg0 + 0));
  b.global_load(9, V(8));
  b.lds_store(V(15), V(9));
  // Btile[ty][tx] = B[t*16 + ty][col]
  b.vadd_i32(10, V(4), S(25));
  b.vshl_b32(10, V(10), Imm(6));
  b.vadd_i32(10, V(10), V(6));
  b.vshl_b32(10, V(10), Imm(2));
  b.vadd_i32(10, V(10), S(KB::kSArg0 + 1));
  b.global_load(11, V(10));
  b.lds_store(V(16), V(11));
  b.barrier();
  for (int k = 0; k < kTile; k++) {
    b.vadd_i32(17, V(14), Imm(static_cast<uint32_t>(k)));
    b.vshl_b32(17, V(17), Imm(2));
    b.lds_load(18, V(17)); // Atile[ty][k]
    b.vadd_i32(19, V(3), Imm(static_cast<uint32_t>(k * kTile)));
    b.vshl_b32(19, V(19), Imm(2));
    b.vadd_i32(19, V(19), Imm(btileBase));
    b.lds_load(21, V(19)); // Btile[k][tx]
    b.vfma_f32(7, V(18), V(21), V(7));
  }
  b.barrier();
  b.sadd_i32(24, S(24), Imm(1));
  b.vcmp_lt_i32(0, S(24), Imm(kMatN / kTile));
  b.endloop(0);
  b.vadd_i32(22, V(13), V(6));
  b.vshl_b32(22, V(22), Imm(2));
  b.vadd_i32(22, V(22), S(KB::kSArg0 + 2));
  b.global_store(V(22), V(7));
  b.end();
  return b.finish();
}

Program buildCopyCoalescedF32() {
  KB b("copy_coalesced_f32");
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4));
  b.vadd_i32(6, V(3), S(KB::kSArg0 + 1));
  b.global_store(V(6), V(5));
  b.end();
  return b.finish();
}

Program buildCopyStridedF32() {
  KB b("copy_strided_f32");
  // Reads in[i * 16], one 4 byte word per 64 byte line, the worst case.
  b.vshl_b32(3, V(KB::kVGlobalTid), Imm(6));
  b.vadd_i32(4, V(3), S(KB::kSArg0 + 0));
  b.global_load(5, V(4));
  b.vshl_b32(6, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(7, V(6), S(KB::kSArg0 + 1));
  b.global_store(V(7), V(5));
  b.end();
  return b.finish();
}

Program buildLdsConflictHeavyI32() {
  KB b("lds_conflict_heavy_i32");
  // Every lane in a 32 lane phase lands in bank 0, full serialization.
  b.vshl_b32(3, V(KB::kVTidInGroup), Imm(7)); // tid * 32 words, in bytes
  b.lds_store(V(3), V(KB::kVGlobalTid));
  b.barrier();
  b.lds_load(4, V(3));
  b.vshl_b32(5, V(KB::kVGlobalTid), Imm(2));
  b.vadd_i32(6, V(5), S(KB::kSArg0 + 0));
  b.global_store(V(6), V(4));
  b.end();
  return b.finish();
}

// ---------------------------------------------------------------- prepares

PreparedKernel prepVecAddI32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "vecadd_i32";
  p.prog = buildVecAdd(false);
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t a = p.mem.alloc(kN * 4), b = p.mem.alloc(kN * 4),
           c = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto va = randomInts(rng, kN, 0, 1u << 20);
  auto vb = randomInts(rng, kN, 0, 1u << 20);
  writeBuf(p.mem, a, va);
  writeBuf(p.mem, b, vb);
  p.args = {a, b, c};
  p.out_base = c;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = static_cast<uint32_t>(static_cast<int32_t>(va[i]) +
                                          static_cast<int32_t>(vb[i]));
  return p;
}

PreparedKernel prepVecAddF32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "vecadd_f32";
  p.prog = buildVecAdd(true);
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t a = p.mem.alloc(kN * 4), b = p.mem.alloc(kN * 4),
           c = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto va = randomFloats(rng, kN);
  auto vb = randomFloats(rng, kN);
  writeBuf(p.mem, a, va);
  writeBuf(p.mem, b, vb);
  p.args = {a, b, c};
  p.out_base = c;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = f32Bits(bitsF32(va[i]) + bitsF32(vb[i]));
  return p;
}

PreparedKernel prepSaxpyF32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "saxpy_f32";
  p.prog = buildSaxpyF32();
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t x = p.mem.alloc(kN * 4), y = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto vx = randomFloats(rng, kN);
  auto vy = randomFloats(rng, kN);
  const float alpha = 1.75f;
  writeBuf(p.mem, x, vx);
  writeBuf(p.mem, y, vy);
  p.args = {x, y, f32Bits(alpha)};
  p.out_base = y;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = f32Bits(std::fmaf(alpha, bitsF32(vx[i]), bitsF32(vy[i])));
  return p;
}

PreparedKernel prepDivergentCollatzI32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "divergent_collatz_i32";
  p.prog = buildDivergentCollatzI32();
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t a = p.mem.alloc(kN * 4), out = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto va = randomInts(rng, kN, 1, 1u << 20);
  writeBuf(p.mem, a, va);
  p.args = {a, out};
  p.out_base = out;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = (va[i] & 1u) ? va[i] * 3u + 1u : va[i] >> 1;
  return p;
}

PreparedKernel prepReduce(bool isFloat, int W, uint32_t seed) {
  PreparedKernel p;
  p.name = isFloat ? "reduce_f32" : "reduce_i32";
  p.prog = buildReduce(isFloat);
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  const uint32_t groups = kN / kGroupSize;
  uint32_t in = p.mem.alloc(kN * 4), out = p.mem.alloc(groups * 4);
  std::mt19937 rng(seed);
  auto vin = isFloat ? randomFloats(rng, kN) : randomInts(rng, kN, 0, 1u << 16);
  writeBuf(p.mem, in, vin);
  p.args = {in, out};
  p.out_base = out;
  p.out_len = groups;
  p.expected.resize(groups);
  // The oracle replays the exact LDS tree order, so floats stay bit exact.
  for (uint32_t g = 0; g < groups; g++) {
    if (isFloat) {
      std::vector<float> buf(kGroupSize);
      for (int i = 0; i < kGroupSize; i++)
        buf[i] = bitsF32(vin[g * kGroupSize + i]);
      for (uint32_t stride = kGroupSize / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; i++)
          buf[i] += buf[i + stride];
      p.expected[g] = f32Bits(buf[0]);
    } else {
      std::vector<uint32_t> buf(vin.begin() + g * kGroupSize,
                                vin.begin() + (g + 1) * kGroupSize);
      for (uint32_t stride = kGroupSize / 2; stride > 0; stride >>= 1)
        for (uint32_t i = 0; i < stride; i++)
          buf[i] += buf[i + stride];
      p.expected[g] = buf[0];
    }
  }
  return p;
}

PreparedKernel prepMatmul(bool tiled, int W, uint32_t seed) {
  PreparedKernel p;
  p.name = tiled ? "matmul_tiled_f32" : "matmul_naive_f32";
  p.prog = tiled ? buildMatmulTiledF32() : buildMatmulNaiveF32();
  p.cfg.wave_width = W;
  p.cfg.workgroup_size = kTile * kTile;
  p.cfg.num_workgroups = (kMatN / kTile) * (kMatN / kTile);
  p.cfg.lds_bytes = kLdsBytes;
  p.mem = GlobalMemory(kMemBytes);
  const uint32_t words = kMatN * kMatN;
  uint32_t a = p.mem.alloc(words * 4), b = p.mem.alloc(words * 4),
           c = p.mem.alloc(words * 4);
  std::mt19937 rng(seed);
  auto va = randomFloats(rng, words);
  auto vb = randomFloats(rng, words);
  writeBuf(p.mem, a, va);
  writeBuf(p.mem, b, vb);
  p.args = {a, b, c};
  p.out_base = c;
  p.out_len = words;
  p.expected.resize(words);
  // Both kernels accumulate in ascending k order with a fused fma.
  for (int row = 0; row < kMatN; row++) {
    for (int col = 0; col < kMatN; col++) {
      float acc = 0.0f;
      for (int k = 0; k < kMatN; k++)
        acc = std::fmaf(bitsF32(va[row * kMatN + k]),
                        bitsF32(vb[k * kMatN + col]), acc);
      p.expected[row * kMatN + col] = f32Bits(acc);
    }
  }
  return p;
}

PreparedKernel prepCopyCoalescedF32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "copy_coalesced_f32";
  p.prog = buildCopyCoalescedF32();
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t in = p.mem.alloc(kN * 4), out = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto vin = randomFloats(rng, kN);
  writeBuf(p.mem, in, vin);
  p.args = {in, out};
  p.out_base = out;
  p.out_len = kN;
  p.expected = vin;
  return p;
}

PreparedKernel prepCopyStridedF32(int W, uint32_t seed) {
  PreparedKernel p;
  p.name = "copy_strided_f32";
  p.prog = buildCopyStridedF32();
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t in = p.mem.alloc(kN * 16 * 4), out = p.mem.alloc(kN * 4);
  std::mt19937 rng(seed);
  auto vin = randomFloats(rng, kN * 16);
  writeBuf(p.mem, in, vin);
  p.args = {in, out};
  p.out_base = out;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = vin[i * 16];
  return p;
}

PreparedKernel prepLdsConflictHeavyI32(int W, uint32_t seed) {
  (void)seed; // outputs depend only on thread ids
  PreparedKernel p;
  p.name = "lds_conflict_heavy_i32";
  p.prog = buildLdsConflictHeavyI32();
  p.cfg = cfg1d(W);
  p.mem = GlobalMemory(kMemBytes);
  uint32_t out = p.mem.alloc(kN * 4);
  p.args = {out};
  p.out_base = out;
  p.out_len = kN;
  p.expected.resize(kN);
  for (uint32_t i = 0; i < kN; i++)
    p.expected[i] = i;
  return p;
}

} // namespace

uint64_t hashWords(const std::vector<uint32_t> &words) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t w : words) {
    for (int i = 0; i < 4; i++) {
      h ^= (w >> (8 * i)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  return h;
}

const std::vector<std::string> &kernelNames() {
  static const std::vector<std::string> kAll = {
      "vecadd_i32",       "vecadd_f32",
      "saxpy_f32",        "divergent_collatz_i32",
      "reduce_i32",       "reduce_f32",
      "matmul_naive_f32", "matmul_tiled_f32",
      "copy_coalesced_f32", "copy_strided_f32",
      "lds_conflict_heavy_i32",
  };
  return kAll;
}

PreparedKernel prepareKernel(const std::string &name, int wave_width,
                             uint32_t seed) {
  if (name == "vecadd_i32")
    return prepVecAddI32(wave_width, seed);
  if (name == "vecadd_f32")
    return prepVecAddF32(wave_width, seed);
  if (name == "saxpy_f32")
    return prepSaxpyF32(wave_width, seed);
  if (name == "divergent_collatz_i32")
    return prepDivergentCollatzI32(wave_width, seed);
  if (name == "reduce_i32")
    return prepReduce(false, wave_width, seed);
  if (name == "reduce_f32")
    return prepReduce(true, wave_width, seed);
  if (name == "matmul_naive_f32")
    return prepMatmul(false, wave_width, seed);
  if (name == "matmul_tiled_f32")
    return prepMatmul(true, wave_width, seed);
  if (name == "copy_coalesced_f32")
    return prepCopyCoalescedF32(wave_width, seed);
  if (name == "copy_strided_f32")
    return prepCopyStridedF32(wave_width, seed);
  if (name == "lds_conflict_heavy_i32")
    return prepLdsConflictHeavyI32(wave_width, seed);
  throw std::runtime_error("unknown kernel: " + name);
}

KernelRunResult runKernel(const std::string &name, int wave_width,
                          uint32_t seed) {
  PreparedKernel p = prepareKernel(name, wave_width, seed);
  Stats st = runProgram(p.prog, p.cfg, p.mem, p.args);
  std::vector<uint32_t> got(p.out_len);
  for (uint32_t i = 0; i < p.out_len; i++)
    got[i] = p.mem.loadWord(p.out_base + i * 4);
  KernelRunResult r;
  r.name = p.name;
  r.stats = st;
  r.oracle_match = got == p.expected;
  r.output_hash = hashWords(got);
  r.elements = p.out_len;
  return r;
}

} // namespace wavesim
