#pragma once

// Small custom RISC-like kernel IR for the wavefront simulator.
// This is not a real GPU ISA and does not follow any real encoding.
// Programs are built through KernelBuilder and interpreted by the simulator.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace wavesim {

enum class Op : uint16_t {
  Nop,

  // Vector ALU, destination is a VGPR, executes per active lane.
  VMov,
  VAddI32,
  VSubI32,
  VMulI32,
  VMinI32,
  VMaxI32,
  VAndB32,
  VOrB32,
  VXorB32,
  VShlB32,
  VShrB32,
  VAddF32,
  VSubF32,
  VMulF32,
  VFmaF32,
  VMinF32,
  VMaxF32,
  VCvtF32I32,
  VCvtI32F32,

  // Vector compares, destination is a mask register.
  // Inactive lanes produce 0 bits.
  VCmpEqI32,
  VCmpNeI32,
  VCmpLtI32,
  VCmpLeI32,
  VCmpGtI32,
  VCmpGeI32,
  VCmpLtF32,
  VCmpGtF32,

  // Scalar ALU, destination is an SGPR, executes once per wavefront.
  SMov,
  SAddI32,
  SSubI32,
  SMulI32,
  SShlB32,
  SShrB32,
  SAndB32,

  // Structured control flow through the divergence stack.
  If,      // a = mask register
  Else,
  EndIf,
  Loop,
  EndLoop, // a = mask register, loop back while any active lane has the bit set

  // Memory. Addresses are byte addresses and must be 4 byte aligned.
  GlobalLoad,  // dst = vgpr, a = address operand
  GlobalStore, // a = address operand, b = data operand
  LdsLoad,     // dst = vgpr, a = address operand
  LdsStore,    // a = address operand, b = data operand

  Barrier, // workgroup barrier, wavefront must be fully converged
  End
};

enum class OpndKind : uint8_t { None, VReg, SReg, MReg, Imm };

struct Operand {
  OpndKind kind = OpndKind::None;
  int reg = 0;
  uint32_t imm = 0;
};

inline Operand V(int r) { return Operand{OpndKind::VReg, r, 0}; }
inline Operand S(int r) { return Operand{OpndKind::SReg, r, 0}; }
inline Operand M(int r) { return Operand{OpndKind::MReg, r, 0}; }
inline Operand Imm(uint32_t v) { return Operand{OpndKind::Imm, 0, v}; }

inline uint32_t f32Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

inline float bitsF32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline Operand FImm(float f) { return Imm(f32Bits(f)); }

struct Instr {
  Op op = Op::Nop;
  Operand dst;
  Operand a;
  Operand b;
  Operand c;
};

struct Program {
  std::string name;
  std::vector<Instr> code;
};

// Convenience builder so kernels read close to pseudocode.
class KernelBuilder {
public:
  explicit KernelBuilder(std::string name) { prog_.name = std::move(name); }

  // Preloaded registers, see sim.cpp for launch conventions.
  static constexpr int kVLane = 0;      // lane id within the wavefront
  static constexpr int kVTidInGroup = 1; // thread id within the workgroup
  static constexpr int kVGlobalTid = 2;  // global thread id
  static constexpr int kSWorkgroupId = 0;
  static constexpr int kSWorkgroupSize = 1;
  static constexpr int kSWaveId = 2;
  static constexpr int kSWaveWidth = 3;
  static constexpr int kSNumWorkgroups = 4;
  static constexpr int kSArg0 = 8; // kernel arguments start here

  void emit(Op op, Operand dst = {}, Operand a = {}, Operand b = {},
            Operand c = {}) {
    prog_.code.push_back(Instr{op, dst, a, b, c});
  }

  void vmov(int d, Operand a) { emit(Op::VMov, V(d), a); }
  void vadd_i32(int d, Operand a, Operand b) { emit(Op::VAddI32, V(d), a, b); }
  void vsub_i32(int d, Operand a, Operand b) { emit(Op::VSubI32, V(d), a, b); }
  void vmul_i32(int d, Operand a, Operand b) { emit(Op::VMulI32, V(d), a, b); }
  void vmin_i32(int d, Operand a, Operand b) { emit(Op::VMinI32, V(d), a, b); }
  void vmax_i32(int d, Operand a, Operand b) { emit(Op::VMaxI32, V(d), a, b); }
  void vand_b32(int d, Operand a, Operand b) { emit(Op::VAndB32, V(d), a, b); }
  void vor_b32(int d, Operand a, Operand b) { emit(Op::VOrB32, V(d), a, b); }
  void vxor_b32(int d, Operand a, Operand b) { emit(Op::VXorB32, V(d), a, b); }
  void vshl_b32(int d, Operand a, Operand b) { emit(Op::VShlB32, V(d), a, b); }
  void vshr_b32(int d, Operand a, Operand b) { emit(Op::VShrB32, V(d), a, b); }
  void vadd_f32(int d, Operand a, Operand b) { emit(Op::VAddF32, V(d), a, b); }
  void vsub_f32(int d, Operand a, Operand b) { emit(Op::VSubF32, V(d), a, b); }
  void vmul_f32(int d, Operand a, Operand b) { emit(Op::VMulF32, V(d), a, b); }
  void vfma_f32(int d, Operand a, Operand b, Operand c) {
    emit(Op::VFmaF32, V(d), a, b, c);
  }
  void vmin_f32(int d, Operand a, Operand b) { emit(Op::VMinF32, V(d), a, b); }
  void vmax_f32(int d, Operand a, Operand b) { emit(Op::VMaxF32, V(d), a, b); }
  void vcvt_f32_i32(int d, Operand a) { emit(Op::VCvtF32I32, V(d), a); }
  void vcvt_i32_f32(int d, Operand a) { emit(Op::VCvtI32F32, V(d), a); }

  void vcmp_eq_i32(int m, Operand a, Operand b) { emit(Op::VCmpEqI32, M(m), a, b); }
  void vcmp_ne_i32(int m, Operand a, Operand b) { emit(Op::VCmpNeI32, M(m), a, b); }
  void vcmp_lt_i32(int m, Operand a, Operand b) { emit(Op::VCmpLtI32, M(m), a, b); }
  void vcmp_le_i32(int m, Operand a, Operand b) { emit(Op::VCmpLeI32, M(m), a, b); }
  void vcmp_gt_i32(int m, Operand a, Operand b) { emit(Op::VCmpGtI32, M(m), a, b); }
  void vcmp_ge_i32(int m, Operand a, Operand b) { emit(Op::VCmpGeI32, M(m), a, b); }
  void vcmp_lt_f32(int m, Operand a, Operand b) { emit(Op::VCmpLtF32, M(m), a, b); }
  void vcmp_gt_f32(int m, Operand a, Operand b) { emit(Op::VCmpGtF32, M(m), a, b); }

  void smov(int d, Operand a) { emit(Op::SMov, S(d), a); }
  void sadd_i32(int d, Operand a, Operand b) { emit(Op::SAddI32, S(d), a, b); }
  void ssub_i32(int d, Operand a, Operand b) { emit(Op::SSubI32, S(d), a, b); }
  void smul_i32(int d, Operand a, Operand b) { emit(Op::SMulI32, S(d), a, b); }
  void sshl_b32(int d, Operand a, Operand b) { emit(Op::SShlB32, S(d), a, b); }
  void sshr_b32(int d, Operand a, Operand b) { emit(Op::SShrB32, S(d), a, b); }
  void sand_b32(int d, Operand a, Operand b) { emit(Op::SAndB32, S(d), a, b); }

  void iff(int m) { emit(Op::If, {}, M(m)); }
  void els() { emit(Op::Else); }
  void endif() { emit(Op::EndIf); }
  void loop() { emit(Op::Loop); }
  void endloop(int m) { emit(Op::EndLoop, {}, M(m)); }

  void global_load(int d, Operand addr) { emit(Op::GlobalLoad, V(d), addr); }
  void global_store(Operand addr, Operand data) {
    emit(Op::GlobalStore, {}, addr, data);
  }
  void lds_load(int d, Operand addr) { emit(Op::LdsLoad, V(d), addr); }
  void lds_store(Operand addr, Operand data) {
    emit(Op::LdsStore, {}, addr, data);
  }
  void barrier() { emit(Op::Barrier); }
  void end() { emit(Op::End); }

  Program finish() { return prog_; }

private:
  Program prog_;
};

} // namespace wavesim
