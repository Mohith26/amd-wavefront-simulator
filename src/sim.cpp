#include "sim.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace wavesim {

void Stats::add(const Stats &o) {
  total_instructions += o.total_instructions;
  vector_instructions += o.vector_instructions;
  divergent_vector_instructions += o.divergent_vector_instructions;
  scalar_instructions += o.scalar_instructions;
  control_instructions += o.control_instructions;
  lds_accesses += o.lds_accesses;
  lds_conflict_cycles += o.lds_conflict_cycles;
  global_accesses += o.global_accesses;
  global_transactions += o.global_transactions;
  barriers += o.barriers;
}

bool Stats::operator==(const Stats &o) const {
  return total_instructions == o.total_instructions &&
         vector_instructions == o.vector_instructions &&
         divergent_vector_instructions == o.divergent_vector_instructions &&
         scalar_instructions == o.scalar_instructions &&
         control_instructions == o.control_instructions &&
         lds_accesses == o.lds_accesses &&
         lds_conflict_cycles == o.lds_conflict_cycles &&
         global_accesses == o.global_accesses &&
         global_transactions == o.global_transactions && barriers == o.barriers;
}

GlobalMemory::GlobalMemory(uint32_t bytes) : words_((bytes + 3) / 4, 0) {}

uint32_t GlobalMemory::alloc(uint32_t bytes) {
  uint32_t addr = (bump_ + 63u) & ~63u; // 64 byte aligned for clean line math
  if (addr + bytes > sizeBytes())
    throw std::runtime_error("global memory allocator out of space");
  bump_ = addr + bytes;
  return addr;
}

uint32_t GlobalMemory::loadWord(uint32_t byteAddr) const {
  if (byteAddr % 4 != 0 || byteAddr + 4 > sizeBytes())
    throw std::runtime_error("global load out of bounds or misaligned");
  return words_[byteAddr / 4];
}

void GlobalMemory::storeWord(uint32_t byteAddr, uint32_t value) {
  if (byteAddr % 4 != 0 || byteAddr + 4 > sizeBytes())
    throw std::runtime_error("global store out of bounds or misaligned");
  words_[byteAddr / 4] = value;
}

namespace {

enum class StackKind : uint8_t { If, Loop };

struct StackEntry {
  StackKind kind;
  uint64_t saved_exec;
  uint64_t cond_mask; // If: mask taken by the then side
  size_t loop_pc;     // Loop: pc of the first body instruction
};

enum class WaveStop { Finished, AtBarrier };

struct Wave {
  int width = 32;
  uint64_t full_mask = 0;
  uint64_t exec = 0;
  uint64_t mreg[kNumMreg] = {};
  uint32_t sgpr[kNumSgpr] = {};
  std::vector<uint32_t> vgpr; // kNumVgpr * width
  size_t pc = 0;
  std::vector<StackEntry> stack;
  bool finished = false;
  bool at_barrier = false;

  uint32_t &v(int reg, int lane) { return vgpr[static_cast<size_t>(reg) * width + lane]; }
};

struct WaveContext {
  GlobalMemory *gmem;
  std::vector<uint32_t> *lds;
  Stats *stats;
};

uint32_t readLane(const Wave &w, const Operand &o, int lane) {
  switch (o.kind) {
  case OpndKind::VReg:
    return w.vgpr[static_cast<size_t>(o.reg) * w.width + lane];
  case OpndKind::SReg:
    return w.sgpr[o.reg];
  case OpndKind::Imm:
    return o.imm;
  default:
    throw std::runtime_error("invalid vector operand");
  }
}

uint32_t readScalar(const Wave &w, const Operand &o) {
  switch (o.kind) {
  case OpndKind::SReg:
    return w.sgpr[o.reg];
  case OpndKind::Imm:
    return o.imm;
  default:
    throw std::runtime_error("invalid scalar operand");
  }
}

bool isVectorOp(Op op) {
  switch (op) {
  case Op::VMov:
  case Op::VAddI32:
  case Op::VSubI32:
  case Op::VMulI32:
  case Op::VMinI32:
  case Op::VMaxI32:
  case Op::VAndB32:
  case Op::VOrB32:
  case Op::VXorB32:
  case Op::VShlB32:
  case Op::VShrB32:
  case Op::VAddF32:
  case Op::VSubF32:
  case Op::VMulF32:
  case Op::VFmaF32:
  case Op::VMinF32:
  case Op::VMaxF32:
  case Op::VCvtF32I32:
  case Op::VCvtI32F32:
  case Op::VCmpEqI32:
  case Op::VCmpNeI32:
  case Op::VCmpLtI32:
  case Op::VCmpLeI32:
  case Op::VCmpGtI32:
  case Op::VCmpGeI32:
  case Op::VCmpLtF32:
  case Op::VCmpGtF32:
  case Op::GlobalLoad:
  case Op::GlobalStore:
  case Op::LdsLoad:
  case Op::LdsStore:
    return true;
  default:
    return false;
  }
}

uint32_t aluVector(Op op, uint32_t ua, uint32_t ub, uint32_t uc) {
  int32_t ia = static_cast<int32_t>(ua);
  int32_t ib = static_cast<int32_t>(ub);
  float fa = bitsF32(ua);
  float fb = bitsF32(ub);
  float fc = bitsF32(uc);
  switch (op) {
  case Op::VMov:
    return ua;
  case Op::VAddI32:
    return static_cast<uint32_t>(ia + ib);
  case Op::VSubI32:
    return static_cast<uint32_t>(ia - ib);
  case Op::VMulI32:
    return static_cast<uint32_t>(ia * ib);
  case Op::VMinI32:
    return static_cast<uint32_t>(std::min(ia, ib));
  case Op::VMaxI32:
    return static_cast<uint32_t>(std::max(ia, ib));
  case Op::VAndB32:
    return ua & ub;
  case Op::VOrB32:
    return ua | ub;
  case Op::VXorB32:
    return ua ^ ub;
  case Op::VShlB32:
    return ua << (ub & 31u);
  case Op::VShrB32:
    return ua >> (ub & 31u);
  case Op::VAddF32:
    return f32Bits(fa + fb);
  case Op::VSubF32:
    return f32Bits(fa - fb);
  case Op::VMulF32:
    return f32Bits(fa * fb);
  case Op::VFmaF32:
    return f32Bits(std::fmaf(fa, fb, fc));
  case Op::VMinF32:
    return f32Bits(std::fmin(fa, fb));
  case Op::VMaxF32:
    return f32Bits(std::fmax(fa, fb));
  case Op::VCvtF32I32:
    return f32Bits(static_cast<float>(ia));
  case Op::VCvtI32F32:
    return static_cast<uint32_t>(static_cast<int32_t>(fa));
  default:
    throw std::runtime_error("not a vector alu op");
  }
}

bool cmpVector(Op op, uint32_t ua, uint32_t ub) {
  int32_t ia = static_cast<int32_t>(ua);
  int32_t ib = static_cast<int32_t>(ub);
  float fa = bitsF32(ua);
  float fb = bitsF32(ub);
  switch (op) {
  case Op::VCmpEqI32:
    return ia == ib;
  case Op::VCmpNeI32:
    return ia != ib;
  case Op::VCmpLtI32:
    return ia < ib;
  case Op::VCmpLeI32:
    return ia <= ib;
  case Op::VCmpGtI32:
    return ia > ib;
  case Op::VCmpGeI32:
    return ia >= ib;
  case Op::VCmpLtF32:
    return fa < fb;
  case Op::VCmpGtF32:
    return fa > fb;
  default:
    throw std::runtime_error("not a compare op");
  }
}

uint32_t aluScalar(Op op, uint32_t a, uint32_t b) {
  int32_t ia = static_cast<int32_t>(a);
  int32_t ib = static_cast<int32_t>(b);
  switch (op) {
  case Op::SMov:
    return a;
  case Op::SAddI32:
    return static_cast<uint32_t>(ia + ib);
  case Op::SSubI32:
    return static_cast<uint32_t>(ia - ib);
  case Op::SMulI32:
    return static_cast<uint32_t>(ia * ib);
  case Op::SShlB32:
    return a << (b & 31u);
  case Op::SShrB32:
    return a >> (b & 31u);
  case Op::SAndB32:
    return a & b;
  default:
    throw std::runtime_error("not a scalar alu op");
  }
}

// LDS and global accesses are processed in 32 lane phases. A wave64 memory
// instruction behaves like two 32 lane accesses for the statistics, which
// mirrors how wide waves are issued in halves on real hardware.
void countLdsPhase(const Wave &w, uint64_t phase_mask,
                   const std::vector<uint32_t> &addrs, Stats &st) {
  if (phase_mask == 0)
    return;
  st.lds_accesses += 1;
  uint32_t bank_words[kLdsBanks][32];
  uint32_t bank_n[kLdsBanks] = {};
  for (int lane = 0; lane < w.width; lane++) {
    if (!(phase_mask >> lane & 1ull))
      continue;
    uint32_t word = addrs[lane] / 4;
    uint32_t bank = word % kLdsBanks;
    bool seen = false;
    for (uint32_t i = 0; i < bank_n[bank]; i++)
      if (bank_words[bank][i] == word) {
        seen = true;
        break;
      }
    if (!seen)
      bank_words[bank][bank_n[bank]++] = word;
  }
  uint32_t worst = 1;
  for (int i = 0; i < kLdsBanks; i++)
    worst = std::max(worst, bank_n[i]);
  st.lds_conflict_cycles += worst - 1;
}

void countGlobalPhase(const Wave &w, uint64_t phase_mask,
                      const std::vector<uint32_t> &addrs, Stats &st) {
  if (phase_mask == 0)
    return;
  st.global_accesses += 1;
  uint32_t lines[32];
  uint32_t n = 0;
  for (int lane = 0; lane < w.width; lane++) {
    if (!(phase_mask >> lane & 1ull))
      continue;
    uint32_t line = addrs[lane] / kCacheLineBytes;
    bool seen = false;
    for (uint32_t i = 0; i < n; i++)
      if (lines[i] == line) {
        seen = true;
        break;
      }
    if (!seen)
      lines[n++] = line;
  }
  st.global_transactions += n;
}

WaveStop stepUntilStop(const Program &prog, Wave &w, WaveContext &ctx) {
  Stats &st = *ctx.stats;
  std::vector<uint32_t> addrs(w.width, 0);
  while (true) {
    if (w.pc >= prog.code.size())
      throw std::runtime_error("pc ran off the end of the program: " + prog.name);
    const Instr &ins = prog.code[w.pc];
    w.pc++;
    st.total_instructions++;

    if (isVectorOp(ins.op)) {
      if (w.exec != 0) {
        st.vector_instructions++;
        if (w.exec != w.full_mask)
          st.divergent_vector_instructions++;
      }
    }

    switch (ins.op) {
    case Op::Nop:
      break;

    case Op::VMov:
    case Op::VAddI32:
    case Op::VSubI32:
    case Op::VMulI32:
    case Op::VMinI32:
    case Op::VMaxI32:
    case Op::VAndB32:
    case Op::VOrB32:
    case Op::VXorB32:
    case Op::VShlB32:
    case Op::VShrB32:
    case Op::VAddF32:
    case Op::VSubF32:
    case Op::VMulF32:
    case Op::VFmaF32:
    case Op::VMinF32:
    case Op::VMaxF32:
    case Op::VCvtF32I32:
    case Op::VCvtI32F32:
      for (int lane = 0; lane < w.width; lane++) {
        if (!(w.exec >> lane & 1ull))
          continue;
        uint32_t ua = readLane(w, ins.a, lane);
        uint32_t ub = ins.b.kind == OpndKind::None ? 0 : readLane(w, ins.b, lane);
        uint32_t uc = ins.c.kind == OpndKind::None ? 0 : readLane(w, ins.c, lane);
        w.v(ins.dst.reg, lane) = aluVector(ins.op, ua, ub, uc);
      }
      break;

    case Op::VCmpEqI32:
    case Op::VCmpNeI32:
    case Op::VCmpLtI32:
    case Op::VCmpLeI32:
    case Op::VCmpGtI32:
    case Op::VCmpGeI32:
    case Op::VCmpLtF32:
    case Op::VCmpGtF32: {
      uint64_t out = 0;
      for (int lane = 0; lane < w.width; lane++) {
        if (!(w.exec >> lane & 1ull))
          continue;
        if (cmpVector(ins.op, readLane(w, ins.a, lane), readLane(w, ins.b, lane)))
          out |= 1ull << lane;
      }
      w.mreg[ins.dst.reg] = out;
      break;
    }

    case Op::SMov:
      st.scalar_instructions++;
      w.sgpr[ins.dst.reg] = readScalar(w, ins.a);
      break;
    case Op::SAddI32:
    case Op::SSubI32:
    case Op::SMulI32:
    case Op::SShlB32:
    case Op::SShrB32:
    case Op::SAndB32:
      st.scalar_instructions++;
      w.sgpr[ins.dst.reg] =
          aluScalar(ins.op, readScalar(w, ins.a), readScalar(w, ins.b));
      break;

    case Op::If: {
      st.control_instructions++;
      StackEntry e;
      e.kind = StackKind::If;
      e.saved_exec = w.exec;
      e.cond_mask = w.exec & w.mreg[ins.a.reg];
      e.loop_pc = 0;
      w.stack.push_back(e);
      w.exec = e.cond_mask;
      break;
    }
    case Op::Else: {
      st.control_instructions++;
      if (w.stack.empty() || w.stack.back().kind != StackKind::If)
        throw std::runtime_error("else without matching if");
      StackEntry &e = w.stack.back();
      w.exec = e.saved_exec & ~e.cond_mask;
      break;
    }
    case Op::EndIf: {
      st.control_instructions++;
      if (w.stack.empty() || w.stack.back().kind != StackKind::If)
        throw std::runtime_error("endif without matching if");
      w.exec = w.stack.back().saved_exec;
      w.stack.pop_back();
      break;
    }
    case Op::Loop: {
      st.control_instructions++;
      StackEntry e;
      e.kind = StackKind::Loop;
      e.saved_exec = w.exec;
      e.cond_mask = 0;
      e.loop_pc = w.pc;
      w.stack.push_back(e);
      break;
    }
    case Op::EndLoop: {
      st.control_instructions++;
      if (w.stack.empty() || w.stack.back().kind != StackKind::Loop)
        throw std::runtime_error("endloop without matching loop");
      StackEntry &e = w.stack.back();
      w.exec &= w.mreg[ins.a.reg];
      if (w.exec != 0) {
        w.pc = e.loop_pc;
      } else {
        w.exec = e.saved_exec;
        w.stack.pop_back();
      }
      break;
    }

    case Op::GlobalLoad:
    case Op::GlobalStore: {
      if (w.exec == 0)
        break;
      for (int lane = 0; lane < w.width; lane++)
        if (w.exec >> lane & 1ull)
          addrs[lane] = readLane(w, ins.a, lane);
      for (int base = 0; base < w.width; base += 32) {
        uint64_t phase = w.exec >> base & 0xffffffffull;
        countGlobalPhase(w, phase << base, addrs, st);
      }
      for (int lane = 0; lane < w.width; lane++) {
        if (!(w.exec >> lane & 1ull))
          continue;
        if (ins.op == Op::GlobalLoad)
          w.v(ins.dst.reg, lane) = ctx.gmem->loadWord(addrs[lane]);
        else
          ctx.gmem->storeWord(addrs[lane], readLane(w, ins.b, lane));
      }
      break;
    }

    case Op::LdsLoad:
    case Op::LdsStore: {
      if (w.exec == 0)
        break;
      std::vector<uint32_t> &lds = *ctx.lds;
      for (int lane = 0; lane < w.width; lane++)
        if (w.exec >> lane & 1ull)
          addrs[lane] = readLane(w, ins.a, lane);
      for (int base = 0; base < w.width; base += 32) {
        uint64_t phase = w.exec >> base & 0xffffffffull;
        countLdsPhase(w, phase << base, addrs, st);
      }
      for (int lane = 0; lane < w.width; lane++) {
        if (!(w.exec >> lane & 1ull))
          continue;
        uint32_t word = addrs[lane] / 4;
        if (addrs[lane] % 4 != 0 || word >= lds.size())
          throw std::runtime_error("lds access out of bounds or misaligned");
        if (ins.op == Op::LdsLoad)
          w.v(ins.dst.reg, lane) = lds[word];
        else
          lds[word] = readLane(w, ins.b, lane);
      }
      break;
    }

    case Op::Barrier:
      st.control_instructions++;
      st.barriers++;
      if (w.exec != w.full_mask)
        throw std::runtime_error("barrier reached with divergent exec mask");
      w.at_barrier = true;
      return WaveStop::AtBarrier;

    case Op::End:
      st.control_instructions++;
      w.finished = true;
      return WaveStop::Finished;
    }
  }
}

} // namespace

Stats runProgram(const Program &prog, const LaunchConfig &cfg, GlobalMemory &mem,
                 const std::vector<uint32_t> &args) {
  if (cfg.wave_width != 32 && cfg.wave_width != 64)
    throw std::runtime_error("wave width must be 32 or 64");
  if (cfg.workgroup_size % cfg.wave_width != 0)
    throw std::runtime_error("workgroup size must be a multiple of wave width");
  if (args.size() > static_cast<size_t>(kNumSgpr - KernelBuilder::kSArg0))
    throw std::runtime_error("too many kernel arguments");

  Stats stats;
  int waves_per_group = cfg.workgroup_size / cfg.wave_width;
  std::vector<uint32_t> lds(cfg.lds_bytes / 4, 0);

  for (int wg = 0; wg < cfg.num_workgroups; wg++) {
    std::fill(lds.begin(), lds.end(), 0u);
    std::vector<Wave> waves(static_cast<size_t>(waves_per_group));
    for (int wv = 0; wv < waves_per_group; wv++) {
      Wave &w = waves[wv];
      w.width = cfg.wave_width;
      w.full_mask = cfg.wave_width == 64 ? ~0ull : (1ull << cfg.wave_width) - 1;
      w.exec = w.full_mask;
      w.vgpr.assign(static_cast<size_t>(kNumVgpr) * cfg.wave_width, 0);
      w.sgpr[KernelBuilder::kSWorkgroupId] = static_cast<uint32_t>(wg);
      w.sgpr[KernelBuilder::kSWorkgroupSize] = static_cast<uint32_t>(cfg.workgroup_size);
      w.sgpr[KernelBuilder::kSWaveId] = static_cast<uint32_t>(wv);
      w.sgpr[KernelBuilder::kSWaveWidth] = static_cast<uint32_t>(cfg.wave_width);
      w.sgpr[KernelBuilder::kSNumWorkgroups] = static_cast<uint32_t>(cfg.num_workgroups);
      for (size_t i = 0; i < args.size(); i++)
        w.sgpr[KernelBuilder::kSArg0 + i] = args[i];
      for (int lane = 0; lane < cfg.wave_width; lane++) {
        uint32_t tid_in_group = static_cast<uint32_t>(wv * cfg.wave_width + lane);
        w.v(KernelBuilder::kVLane, lane) = static_cast<uint32_t>(lane);
        w.v(KernelBuilder::kVTidInGroup, lane) = tid_in_group;
        w.v(KernelBuilder::kVGlobalTid, lane) =
            static_cast<uint32_t>(wg) * cfg.workgroup_size + tid_in_group;
      }
    }

    WaveContext ctx{&mem, &lds, &stats};
    while (true) {
      bool all_finished = true;
      for (auto &w : waves) {
        if (w.finished || w.at_barrier)
          continue;
        stepUntilStop(prog, w, ctx);
      }
      for (auto &w : waves)
        if (!w.finished)
          all_finished = false;
      if (all_finished)
        break;
      // Every unfinished wave must be parked at the barrier now.
      for (auto &w : waves) {
        if (w.finished)
          continue;
        if (!w.at_barrier)
          throw std::runtime_error("barrier deadlock: wave stopped without barrier");
        w.at_barrier = false;
      }
    }
  }
  return stats;
}

} // namespace wavesim
