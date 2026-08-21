// Assert style test harness for the simulator. Run with make test.
// Micro programs verify instruction semantics, the divergence stack,
// barriers, and the hand computed LDS and coalescing cases. Kernel tests
// verify every kernel bit exact against its scalar oracle.

#include "../kernels/kernels.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace wavesim;
using KB = KernelBuilder;

static int g_tests = 0;
static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    g_checks++;                                                                \
    if (!(cond)) {                                                             \
      g_failures++;                                                            \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                          \
  } while (0)

static void runTest(const char *name, const std::function<void()> &fn) {
  g_tests++;
  std::printf("[%02d] %s\n", g_tests, name);
  fn();
}

namespace {

struct Mini {
  GlobalMemory mem{1u << 20};
  uint32_t out = 0;
  Stats st;
};

Mini runMini(const Program &p, int wave_width = 32, int wg_size = -1,
             int num_wg = 1,
             const std::function<void(GlobalMemory &, uint32_t)> &init = {}) {
  Mini m;
  m.out = m.mem.alloc(64 * 1024);
  if (init)
    init(m.mem, m.out);
  LaunchConfig cfg;
  cfg.wave_width = wave_width;
  cfg.workgroup_size = wg_size < 0 ? wave_width : wg_size;
  cfg.num_workgroups = num_wg;
  cfg.lds_bytes = 32 * 1024;
  m.st = runProgram(p, cfg, m.mem, {m.out});
  return m;
}

uint32_t word(const Mini &m, uint32_t idx) {
  return m.mem.loadWord(m.out + idx * 4);
}

// Emits: store value operand at out[slot * 32 + lane].
void storeSlot(KB &b, int slot, Operand value) {
  b.vshl_b32(30, V(KB::kVLane), Imm(2));
  b.vadd_i32(30, V(30), S(KB::kSArg0));
  b.vadd_i32(30, V(30), Imm(static_cast<uint32_t>(slot) * 128));
  b.global_store(V(30), value);
}

bool throwsRuntimeError(const std::function<void()> &fn) {
  try {
    fn();
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

// ------------------------------------------------------------ ALU semantics

void testVectorIntOps() {
  KB b("valu_int");
  b.vadd_i32(5, V(0), Imm(7));
  storeSlot(b, 0, V(5));
  b.vsub_i32(5, V(0), Imm(3));
  storeSlot(b, 1, V(5));
  b.vmul_i32(5, V(0), Imm(5));
  storeSlot(b, 2, V(5));
  b.vmin_i32(5, V(0), Imm(10));
  storeSlot(b, 3, V(5));
  b.vmax_i32(5, V(0), Imm(10));
  storeSlot(b, 4, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    int32_t L = lane;
    CHECK(word(m, 0 * 32 + lane) == static_cast<uint32_t>(L + 7));
    CHECK(word(m, 1 * 32 + lane) == static_cast<uint32_t>(L - 3));
    CHECK(word(m, 2 * 32 + lane) == static_cast<uint32_t>(L * 5));
    CHECK(word(m, 3 * 32 + lane) == static_cast<uint32_t>(std::min(L, 10)));
    CHECK(word(m, 4 * 32 + lane) == static_cast<uint32_t>(std::max(L, 10)));
  }
}

void testVectorBitOps() {
  KB b("valu_bits");
  b.vand_b32(5, V(0), Imm(6));
  storeSlot(b, 0, V(5));
  b.vor_b32(5, V(0), Imm(6));
  storeSlot(b, 1, V(5));
  b.vxor_b32(5, V(0), Imm(6));
  storeSlot(b, 2, V(5));
  b.vshl_b32(5, V(0), Imm(3));
  storeSlot(b, 3, V(5));
  b.vshr_b32(5, V(0), Imm(1));
  storeSlot(b, 4, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (uint32_t lane = 0; lane < 32; lane++) {
    CHECK(word(m, 0 * 32 + lane) == (lane & 6u));
    CHECK(word(m, 1 * 32 + lane) == (lane | 6u));
    CHECK(word(m, 2 * 32 + lane) == (lane ^ 6u));
    CHECK(word(m, 3 * 32 + lane) == (lane << 3));
    CHECK(word(m, 4 * 32 + lane) == (lane >> 1));
  }
}

void testVectorFloatOps() {
  KB b("valu_float");
  b.vcvt_f32_i32(4, V(0)); // f = float(lane)
  b.vadd_f32(5, V(4), FImm(1.25f));
  storeSlot(b, 0, V(5));
  b.vsub_f32(5, V(4), FImm(8.5f));
  storeSlot(b, 1, V(5));
  b.vmul_f32(5, V(4), FImm(0.375f));
  storeSlot(b, 2, V(5));
  b.vfma_f32(5, V(4), FImm(2.0f), FImm(0.5f));
  storeSlot(b, 3, V(5));
  b.vmin_f32(5, V(4), FImm(7.0f));
  storeSlot(b, 4, V(5));
  b.vmax_f32(5, V(4), FImm(7.0f));
  storeSlot(b, 5, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    float f = static_cast<float>(lane);
    CHECK(word(m, 0 * 32 + lane) == f32Bits(f + 1.25f));
    CHECK(word(m, 1 * 32 + lane) == f32Bits(f - 8.5f));
    CHECK(word(m, 2 * 32 + lane) == f32Bits(f * 0.375f));
    CHECK(word(m, 3 * 32 + lane) == f32Bits(std::fmaf(f, 2.0f, 0.5f)));
    CHECK(word(m, 4 * 32 + lane) == f32Bits(std::fmin(f, 7.0f)));
    CHECK(word(m, 5 * 32 + lane) == f32Bits(std::fmax(f, 7.0f)));
  }
}

void testConvertOps() {
  KB b("valu_cvt");
  b.vcvt_f32_i32(4, V(0));
  storeSlot(b, 0, V(4));
  b.vcvt_i32_f32(5, V(4));
  storeSlot(b, 1, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (uint32_t lane = 0; lane < 32; lane++) {
    CHECK(word(m, 0 * 32 + lane) == f32Bits(static_cast<float>(lane)));
    CHECK(word(m, 1 * 32 + lane) == lane);
  }
}

void testScalarOps() {
  KB b("salu");
  b.smov(20, Imm(5));
  storeSlot(b, 0, S(20));
  b.sadd_i32(20, S(20), Imm(9)); // 14
  storeSlot(b, 1, S(20));
  b.ssub_i32(20, S(20), Imm(4)); // 10
  storeSlot(b, 2, S(20));
  b.smul_i32(20, S(20), Imm(3)); // 30
  storeSlot(b, 3, S(20));
  b.sshl_b32(20, S(20), Imm(1)); // 60
  storeSlot(b, 4, S(20));
  b.sshr_b32(20, S(20), Imm(2)); // 15
  storeSlot(b, 5, S(20));
  b.sand_b32(20, S(20), Imm(9)); // 9
  storeSlot(b, 6, S(20));
  b.end();
  Mini m = runMini(b.finish());
  const uint32_t expect[7] = {5, 14, 10, 30, 60, 15, 9};
  for (int s = 0; s < 7; s++)
    for (int lane = 0; lane < 32; lane++)
      CHECK(word(m, static_cast<uint32_t>(s) * 32 + lane) == expect[s]);
}

// ------------------------------------------------------------- compare ops

void cmpCase(const char *name, void (KB::*emit)(int, Operand, Operand),
             const std::function<bool(int)> &host) {
  (void)name;
  KB b("cmp_case");
  (b.*emit)(0, V(KB::kVLane), Imm(16));
  b.iff(0);
  b.vmov(5, Imm(1));
  b.els();
  b.vmov(5, Imm(0));
  b.endif();
  storeSlot(b, 0, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++)
    CHECK(word(m, lane) == (host(lane) ? 1u : 0u));
}

void testCmpEq() {
  cmpCase("eq", &KB::vcmp_eq_i32, [](int l) { return l == 16; });
}
void testCmpNe() {
  cmpCase("ne", &KB::vcmp_ne_i32, [](int l) { return l != 16; });
}
void testCmpLt() {
  cmpCase("lt", &KB::vcmp_lt_i32, [](int l) { return l < 16; });
}
void testCmpLe() {
  cmpCase("le", &KB::vcmp_le_i32, [](int l) { return l <= 16; });
}
void testCmpGt() {
  cmpCase("gt", &KB::vcmp_gt_i32, [](int l) { return l > 16; });
}
void testCmpGe() {
  cmpCase("ge", &KB::vcmp_ge_i32, [](int l) { return l >= 16; });
}

void testCmpFloat() {
  KB b("cmp_float");
  b.vsub_i32(3, V(0), Imm(16));
  b.vcvt_f32_i32(4, V(3)); // lane - 16 as float
  b.vcmp_lt_f32(0, V(4), FImm(0.5f));
  b.iff(0);
  b.vmov(5, Imm(1));
  b.els();
  b.vmov(5, Imm(0));
  b.endif();
  storeSlot(b, 0, V(5));
  b.vcmp_gt_f32(1, V(4), FImm(-3.0f));
  b.iff(1);
  b.vmov(6, Imm(1));
  b.els();
  b.vmov(6, Imm(0));
  b.endif();
  storeSlot(b, 1, V(6));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    float f = static_cast<float>(lane - 16);
    CHECK(word(m, 0 * 32 + lane) == (f < 0.5f ? 1u : 0u));
    CHECK(word(m, 1 * 32 + lane) == (f > -3.0f ? 1u : 0u));
  }
}

// --------------------------------------------------- divergence stack tests

void testIfElsePartition() {
  KB b("if_else");
  b.vcmp_lt_i32(0, V(0), Imm(16));
  b.iff(0);
  b.vmov(5, Imm(111));
  b.els();
  b.vmov(5, Imm(222));
  b.endif();
  storeSlot(b, 0, V(5));
  b.vmov(6, Imm(99)); // exec must be fully restored here
  storeSlot(b, 1, V(6));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    CHECK(word(m, lane) == (lane < 16 ? 111u : 222u));
    CHECK(word(m, 32 + lane) == 99u);
  }
}

void testNestedIf() {
  KB b("nested_if");
  b.vand_b32(3, V(0), Imm(1));
  b.vcmp_eq_i32(0, V(3), Imm(1)); // odd lanes
  b.vand_b32(4, V(0), Imm(2));
  b.vcmp_eq_i32(1, V(4), Imm(2)); // bit 1 set
  b.vmov(5, Imm(77));
  b.iff(0);
  b.iff(1);
  b.vmov(5, Imm(3));
  b.els();
  b.vmov(5, Imm(1));
  b.endif();
  b.els();
  b.iff(1);
  b.vmov(5, Imm(2));
  b.els();
  b.vmov(5, Imm(0));
  b.endif();
  b.endif();
  storeSlot(b, 0, V(5));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    uint32_t expect = (lane & 1 ? (lane & 2 ? 3u : 1u) : (lane & 2 ? 2u : 0u));
    CHECK(word(m, lane) == expect);
  }
}

void testIfAllFalse() {
  KB b("if_all_false");
  b.vcmp_lt_i32(0, V(0), Imm(0)); // false everywhere
  b.iff(0);
  storeSlot(b, 0, Imm(123));
  b.endif();
  storeSlot(b, 1, Imm(55));
  b.end();
  Mini m = runMini(b.finish(), 32, -1, 1,
                   [](GlobalMemory &mem, uint32_t out) {
                     for (int i = 0; i < 64; i++)
                       mem.storeWord(out + i * 4, 7u);
                   });
  for (int lane = 0; lane < 32; lane++) {
    CHECK(word(m, lane) == 7u); // untouched, store was fully masked
    CHECK(word(m, 32 + lane) == 55u);
  }
  // The masked store must not count as a memory access either.
  CHECK(m.st.global_accesses == 1);
}

void testLoopCountdown() {
  KB b("loop_countdown");
  b.vmov(3, V(0)); // counter = lane
  b.vmov(4, Imm(0));
  b.loop();
  b.vadd_i32(4, V(4), Imm(1));
  b.vsub_i32(3, V(3), Imm(1));
  b.vcmp_gt_i32(0, V(3), Imm(0));
  b.endloop(0);
  storeSlot(b, 0, V(4));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++) {
    uint32_t expect = lane > 1 ? static_cast<uint32_t>(lane) : 1u;
    CHECK(word(m, lane) == expect);
  }
}

void testLoopTriangular() {
  KB b("loop_triangular");
  b.vmov(3, V(0)); // counter = lane
  b.vmov(4, Imm(0));
  b.loop();
  b.vadd_i32(4, V(4), V(3));
  b.vsub_i32(3, V(3), Imm(1));
  b.vcmp_gt_i32(0, V(3), Imm(0));
  b.endloop(0);
  storeSlot(b, 0, V(4));
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++)
    CHECK(word(m, lane) == static_cast<uint32_t>(lane * (lane + 1) / 2));
}

void testIfInsideLoop() {
  KB b("if_inside_loop");
  b.vmov(3, V(0)); // counter = lane
  b.vmov(4, Imm(0));
  b.loop();
  b.vand_b32(5, V(3), Imm(1));
  b.vcmp_eq_i32(1, V(5), Imm(1));
  b.iff(1);
  b.vadd_i32(4, V(4), Imm(1));
  b.endif();
  b.vsub_i32(3, V(3), Imm(1));
  b.vcmp_gt_i32(0, V(3), Imm(0));
  b.endloop(0);
  storeSlot(b, 0, V(4)); // number of odd values in lane..1
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++)
    CHECK(word(m, lane) == static_cast<uint32_t>((lane + 1) / 2));
}

void testExecRestoredAfterLoop() {
  KB b("exec_after_loop");
  b.vmov(3, V(0));
  b.loop();
  b.vsub_i32(3, V(3), Imm(1));
  b.vcmp_gt_i32(0, V(3), Imm(0));
  b.endloop(0);
  storeSlot(b, 0, Imm(31337)); // must reach every lane
  b.end();
  Mini m = runMini(b.finish());
  for (int lane = 0; lane < 32; lane++)
    CHECK(word(m, lane) == 31337u);
}

void testDivergenceCounters() {
  KernelRunResult flat = runKernel("vecadd_i32", 32, 1);
  CHECK(flat.stats.divergent_vector_instructions == 0);
  KernelRunResult div = runKernel("divergent_collatz_i32", 32, 1);
  CHECK(div.stats.divergent_vector_instructions > 0);
  CHECK(div.stats.divergent_vector_instructions < div.stats.vector_instructions);
}

// ------------------------------------------------------------ barrier tests

void testBarrierCrossWave() {
  KB b("barrier_cross_wave");
  b.vshl_b32(3, V(KB::kVTidInGroup), Imm(2));
  b.vmul_i32(4, V(KB::kVTidInGroup), Imm(10));
  b.lds_store(V(3), V(4));
  b.barrier();
  b.vsub_i32(5, Imm(63), V(KB::kVTidInGroup));
  b.vshl_b32(6, V(5), Imm(2));
  b.lds_load(7, V(6));
  b.vadd_i32(8, V(3), S(KB::kSArg0));
  b.global_store(V(8), V(7));
  b.end();
  Mini m = runMini(b.finish(), 32, 64, 1);
  for (int tid = 0; tid < 64; tid++)
    CHECK(word(m, tid) == static_cast<uint32_t>((63 - tid) * 10));
}

void testBarrierCountsReduce() {
  KernelRunResult r = runKernel("reduce_i32", 32, 3);
  // 16 workgroups, 8 waves each, 1 barrier after the LDS fill plus 8 loop
  // iterations with one barrier each.
  CHECK(r.stats.barriers == 16u * 8u * 9u);
  KernelRunResult r64 = runKernel("reduce_i32", 64, 3);
  CHECK(r64.stats.barriers == 16u * 4u * 9u);
}

void testBarrierDivergentThrows() {
  KB b("barrier_divergent");
  b.vcmp_lt_i32(0, V(0), Imm(16));
  b.iff(0);
  b.barrier();
  b.endif();
  b.end();
  Program p = b.finish();
  CHECK(throwsRuntimeError([&] { runMini(p); }));
}

void testGlobalOobThrows() {
  KB b("global_oob");
  b.global_load(3, Imm(1u << 30));
  b.end();
  Program p = b.finish();
  CHECK(throwsRuntimeError([&] { runMini(p); }));
}

void testLdsOobThrows() {
  KB b("lds_oob");
  b.lds_load(3, Imm(1u << 20));
  b.end();
  Program p = b.finish();
  CHECK(throwsRuntimeError([&] { runMini(p); }));
}

// ----------------------------------------- hand computed LDS conflict cases

Stats ldsPattern(int shift, int wave_width) {
  KB b("lds_pattern");
  b.vshl_b32(3, V(KB::kVLane), Imm(static_cast<uint32_t>(shift)));
  b.lds_store(V(3), V(KB::kVLane));
  b.end();
  return runMini(b.finish(), wave_width).st;
}

void testLdsStride1() {
  Stats st = ldsPattern(2, 32); // word stride 1, every lane its own bank
  CHECK(st.lds_accesses == 1);
  CHECK(st.lds_conflict_cycles == 0);
}

void testLdsStride2() {
  Stats st = ldsPattern(3, 32); // word stride 2, two lanes per bank
  CHECK(st.lds_accesses == 1);
  CHECK(st.lds_conflict_cycles == 1);
}

void testLdsStride32() {
  Stats st = ldsPattern(7, 32); // word stride 32, all lanes in bank 0
  CHECK(st.lds_accesses == 1);
  CHECK(st.lds_conflict_cycles == 31);
}

void testLdsBroadcast() {
  KB b("lds_broadcast");
  b.lds_store(Imm(128), V(KB::kVLane));
  b.lds_load(4, Imm(128));
  b.end();
  Stats st = runMini(b.finish()).st;
  CHECK(st.lds_accesses == 2);
  CHECK(st.lds_conflict_cycles == 0); // same address is a broadcast
}

void testLdsPartialMask() {
  KB b("lds_partial");
  b.vcmp_lt_i32(0, V(0), Imm(8));
  b.iff(0);
  b.vshl_b32(3, V(0), Imm(7)); // word stride 32, but only 8 active lanes
  b.lds_store(V(3), V(0));
  b.endif();
  b.end();
  Stats st = runMini(b.finish()).st;
  CHECK(st.lds_accesses == 1);
  CHECK(st.lds_conflict_cycles == 7);
}

void testLdsWave64Stride1() {
  Stats st = ldsPattern(2, 64); // two 32 lane phases, both conflict free
  CHECK(st.lds_accesses == 2);
  CHECK(st.lds_conflict_cycles == 0);
}

void testLdsWave64Stride32() {
  Stats st = ldsPattern(7, 64); // both phases fully serialized in bank 0
  CHECK(st.lds_accesses == 2);
  CHECK(st.lds_conflict_cycles == 62);
}

// -------------------------------------- hand computed coalescing cases

Stats globalPattern(int shift, int wave_width) {
  KB b("global_pattern");
  b.vshl_b32(3, V(KB::kVLane), Imm(static_cast<uint32_t>(shift)));
  b.vadd_i32(3, V(3), S(KB::kSArg0));
  b.global_load(4, V(3));
  b.end();
  return runMini(b.finish(), wave_width).st;
}

void testGlobalContiguous() {
  Stats st = globalPattern(2, 32); // 128 contiguous bytes, 64 byte lines
  CHECK(st.global_accesses == 1);
  CHECK(st.global_transactions == 2);
}

void testGlobalStride64() {
  Stats st = globalPattern(6, 32); // one word per line, worst case
  CHECK(st.global_accesses == 1);
  CHECK(st.global_transactions == 32);
}

void testGlobalStride8() {
  Stats st = globalPattern(3, 32); // 256 bytes touched, 4 lines
  CHECK(st.global_accesses == 1);
  CHECK(st.global_transactions == 4);
}

void testGlobalBroadcast() {
  KB b("global_broadcast");
  b.global_load(4, S(KB::kSArg0)); // every lane reads the same address
  b.end();
  Stats st = runMini(b.finish()).st;
  CHECK(st.global_accesses == 1);
  CHECK(st.global_transactions == 1);
}

void testGlobalPartialMask() {
  KB b("global_partial");
  b.vcmp_lt_i32(0, V(0), Imm(8));
  b.iff(0);
  b.vshl_b32(3, V(0), Imm(2));
  b.vadd_i32(3, V(3), S(KB::kSArg0));
  b.global_load(4, V(3)); // 32 bytes inside one aligned line
  b.endif();
  b.end();
  Stats st = runMini(b.finish()).st;
  CHECK(st.global_accesses == 1);
  CHECK(st.global_transactions == 1);
}

void testGlobalWave64Contiguous() {
  Stats st = globalPattern(2, 64); // two phases of 128 bytes each
  CHECK(st.global_accesses == 2);
  CHECK(st.global_transactions == 4);
}

// ------------------------------------------------------- kernels vs oracle

void oracleCase(const char *name) {
  KernelRunResult r = runKernel(name, 32, 42);
  CHECK(r.oracle_match);
}

void testKernelVecAddI32() { oracleCase("vecadd_i32"); }
void testKernelVecAddF32() { oracleCase("vecadd_f32"); }
void testKernelSaxpy() { oracleCase("saxpy_f32"); }
void testKernelCollatz() { oracleCase("divergent_collatz_i32"); }
void testKernelReduceI32() { oracleCase("reduce_i32"); }
void testKernelReduceF32() { oracleCase("reduce_f32"); }
void testKernelMatmulNaive() { oracleCase("matmul_naive_f32"); }
void testKernelMatmulTiled() { oracleCase("matmul_tiled_f32"); }
void testKernelCopyCoalesced() { oracleCase("copy_coalesced_f32"); }
void testKernelCopyStrided() { oracleCase("copy_strided_f32"); }
void testKernelLdsConflictHeavy() { oracleCase("lds_conflict_heavy_i32"); }

// ------------------------------------------------- wave32 vs wave64 results

void equivalenceCase(const char *name) {
  KernelRunResult a = runKernel(name, 32, 42);
  KernelRunResult c = runKernel(name, 64, 42);
  CHECK(a.oracle_match);
  CHECK(c.oracle_match);
  CHECK(a.output_hash == c.output_hash);
}

void testWave64VecAdd() { equivalenceCase("vecadd_f32"); }
void testWave64ReduceF32() { equivalenceCase("reduce_f32"); }
void testWave64MatmulTiled() { equivalenceCase("matmul_tiled_f32"); }
void testWave64Collatz() { equivalenceCase("divergent_collatz_i32"); }

// ------------------------------------------------------------- determinism

void testDeterminismSameSeed() {
  KernelRunResult a = runKernel("saxpy_f32", 32, 7);
  KernelRunResult c = runKernel("saxpy_f32", 32, 7);
  CHECK(a.output_hash == c.output_hash);
  CHECK(a.stats == c.stats);
  KernelRunResult d = runKernel("matmul_tiled_f32", 32, 7);
  KernelRunResult e = runKernel("matmul_tiled_f32", 32, 7);
  CHECK(d.output_hash == e.output_hash);
  CHECK(d.stats == e.stats);
}

void testDeterminismDifferentSeed() {
  KernelRunResult a = runKernel("saxpy_f32", 32, 7);
  KernelRunResult c = runKernel("saxpy_f32", 32, 8);
  CHECK(a.output_hash != c.output_hash);
}

// ------------------------------------------------------ whole kernel stats

void testCopyKernelStats() {
  // 4096 threads in 128 wave32 phases for the load plus 128 for the store.
  KernelRunResult co = runKernel("copy_coalesced_f32", 32, 42);
  CHECK(co.stats.global_accesses == 256);
  CHECK(co.stats.global_transactions == 512); // 2 lines per phase
  KernelRunResult st = runKernel("copy_strided_f32", 32, 42);
  CHECK(st.stats.global_accesses == 256);
  // 32 lines per strided load phase, 2 per coalesced store phase.
  CHECK(st.stats.global_transactions == 128u * 32u + 128u * 2u);
}

void testLdsConflictHeavyStats() {
  KernelRunResult r = runKernel("lds_conflict_heavy_i32", 32, 42);
  // One store plus one load phase per wave, 128 waves, 31 extra cycles each.
  CHECK(r.stats.lds_accesses == 256);
  CHECK(r.stats.lds_conflict_cycles == 256u * 31u);
}

void testMatmulLdsContrast() {
  KernelRunResult tiled = runKernel("matmul_tiled_f32", 32, 42);
  CHECK(tiled.stats.lds_accesses > 0);
  CHECK(tiled.stats.lds_conflict_cycles == 0); // tile layout is conflict free
  KernelRunResult naive = runKernel("matmul_naive_f32", 32, 42);
  CHECK(naive.stats.lds_accesses == 0);
  CHECK(naive.stats.global_transactions > tiled.stats.global_transactions);
}

} // namespace

int main() {
  runTest("vector int alu ops", testVectorIntOps);
  runTest("vector bit ops", testVectorBitOps);
  runTest("vector float alu ops", testVectorFloatOps);
  runTest("int float conversions", testConvertOps);
  runTest("scalar alu ops", testScalarOps);
  runTest("cmp eq i32", testCmpEq);
  runTest("cmp ne i32", testCmpNe);
  runTest("cmp lt i32", testCmpLt);
  runTest("cmp le i32", testCmpLe);
  runTest("cmp gt i32", testCmpGt);
  runTest("cmp ge i32", testCmpGe);
  runTest("cmp f32", testCmpFloat);
  runTest("if else partition", testIfElsePartition);
  runTest("nested if", testNestedIf);
  runTest("if with all false condition", testIfAllFalse);
  runTest("loop countdown per lane", testLoopCountdown);
  runTest("loop triangular sums", testLoopTriangular);
  runTest("if inside loop", testIfInsideLoop);
  runTest("exec restored after loop", testExecRestoredAfterLoop);
  runTest("divergence counters", testDivergenceCounters);
  runTest("barrier cross wave communication", testBarrierCrossWave);
  runTest("barrier counts in reduction", testBarrierCountsReduce);
  runTest("barrier under divergence throws", testBarrierDivergentThrows);
  runTest("global out of bounds throws", testGlobalOobThrows);
  runTest("lds out of bounds throws", testLdsOobThrows);
  runTest("lds stride 1 conflict free", testLdsStride1);
  runTest("lds stride 2 two way conflict", testLdsStride2);
  runTest("lds stride 32 full serialization", testLdsStride32);
  runTest("lds broadcast no conflict", testLdsBroadcast);
  runTest("lds conflicts under partial mask", testLdsPartialMask);
  runTest("lds wave64 stride 1", testLdsWave64Stride1);
  runTest("lds wave64 stride 32", testLdsWave64Stride32);
  runTest("global contiguous coalesced", testGlobalContiguous);
  runTest("global stride 64B uncoalesced", testGlobalStride64);
  runTest("global stride 8B partial coalescing", testGlobalStride8);
  runTest("global broadcast single line", testGlobalBroadcast);
  runTest("global access under partial mask", testGlobalPartialMask);
  runTest("global wave64 contiguous", testGlobalWave64Contiguous);
  runTest("kernel vecadd_i32 vs oracle", testKernelVecAddI32);
  runTest("kernel vecadd_f32 vs oracle", testKernelVecAddF32);
  runTest("kernel saxpy_f32 vs oracle", testKernelSaxpy);
  runTest("kernel divergent_collatz_i32 vs oracle", testKernelCollatz);
  runTest("kernel reduce_i32 vs oracle", testKernelReduceI32);
  runTest("kernel reduce_f32 vs oracle", testKernelReduceF32);
  runTest("kernel matmul_naive_f32 vs oracle", testKernelMatmulNaive);
  runTest("kernel matmul_tiled_f32 vs oracle", testKernelMatmulTiled);
  runTest("kernel copy_coalesced_f32 vs oracle", testKernelCopyCoalesced);
  runTest("kernel copy_strided_f32 vs oracle", testKernelCopyStrided);
  runTest("kernel lds_conflict_heavy_i32 vs oracle", testKernelLdsConflictHeavy);
  runTest("wave64 equivalence vecadd_f32", testWave64VecAdd);
  runTest("wave64 equivalence reduce_f32", testWave64ReduceF32);
  runTest("wave64 equivalence matmul_tiled_f32", testWave64MatmulTiled);
  runTest("wave64 equivalence divergent kernel", testWave64Collatz);
  runTest("determinism with same seed", testDeterminismSameSeed);
  runTest("determinism with different seed", testDeterminismDifferentSeed);
  runTest("copy kernel transaction totals", testCopyKernelStats);
  runTest("lds conflict heavy kernel totals", testLdsConflictHeavyStats);
  runTest("tiled vs naive matmul lds contrast", testMatmulLdsContrast);

  std::printf("\n%d tests, %d checks, %d failures\n", g_tests, g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
