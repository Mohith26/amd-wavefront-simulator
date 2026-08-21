// Benchmark and stats reporter. Run with make bench.
// Writes results/results.json. Instruction throughput numbers time only the
// simulator launch, not input setup or oracle computation. All statistics
// come from the counters in the simulator, this is not a timing model.

#include "../kernels/kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace wavesim;
using KB = KernelBuilder;

namespace {

double now() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct BenchRow {
  KernelRunResult res;
  double instr_per_sec = 0.0;
  int reps = 0;
  double seconds = 0.0;
};

BenchRow benchKernel(const std::string &name, int wave_width, uint32_t seed) {
  BenchRow row;
  row.res = runKernel(name, wave_width, seed);

  PreparedKernel p = prepareKernel(name, wave_width, seed);
  runProgram(p.prog, p.cfg, p.mem, p.args); // warmup
  double t0 = now();
  runProgram(p.prog, p.cfg, p.mem, p.args);
  double once = now() - t0;
  int reps = static_cast<int>(std::ceil(0.5 / std::max(once, 1e-6)));
  if (reps < 3)
    reps = 3;
  if (reps > 2000)
    reps = 2000;
  double t1 = now();
  for (int i = 0; i < reps; i++)
    runProgram(p.prog, p.cfg, p.mem, p.args);
  double elapsed = now() - t1;
  row.reps = reps;
  row.seconds = elapsed;
  row.instr_per_sec =
      static_cast<double>(row.res.stats.total_instructions) * reps / elapsed;
  return row;
}

double divergencePct(const Stats &st) {
  if (st.vector_instructions == 0)
    return 0.0;
  return 100.0 * static_cast<double>(st.divergent_vector_instructions) /
         static_cast<double>(st.vector_instructions);
}

double transPerAccess(const Stats &st) {
  if (st.global_accesses == 0)
    return 0.0;
  return static_cast<double>(st.global_transactions) /
         static_cast<double>(st.global_accesses);
}

void printRowJson(FILE *f, const BenchRow &row, bool last) {
  const Stats &st = row.res.stats;
  std::fprintf(f, "    {\n");
  std::fprintf(f, "      \"kernel\": \"%s\",\n", row.res.name.c_str());
  std::fprintf(f, "      \"oracle_bit_exact\": %s,\n",
               row.res.oracle_match ? "true" : "false");
  std::fprintf(f, "      \"output_words\": %u,\n", row.res.elements);
  std::fprintf(f, "      \"output_fnv1a\": \"%016llx\",\n",
               static_cast<unsigned long long>(row.res.output_hash));
  std::fprintf(f, "      \"instructions\": %llu,\n",
               static_cast<unsigned long long>(st.total_instructions));
  std::fprintf(f, "      \"vector_instructions\": %llu,\n",
               static_cast<unsigned long long>(st.vector_instructions));
  std::fprintf(f, "      \"scalar_instructions\": %llu,\n",
               static_cast<unsigned long long>(st.scalar_instructions));
  std::fprintf(f, "      \"divergent_vector_instructions\": %llu,\n",
               static_cast<unsigned long long>(st.divergent_vector_instructions));
  std::fprintf(f, "      \"divergence_pct\": %.4f,\n", divergencePct(st));
  std::fprintf(f, "      \"lds_accesses\": %llu,\n",
               static_cast<unsigned long long>(st.lds_accesses));
  std::fprintf(f, "      \"lds_conflict_cycles\": %llu,\n",
               static_cast<unsigned long long>(st.lds_conflict_cycles));
  std::fprintf(f, "      \"global_accesses\": %llu,\n",
               static_cast<unsigned long long>(st.global_accesses));
  std::fprintf(f, "      \"global_transactions\": %llu,\n",
               static_cast<unsigned long long>(st.global_transactions));
  std::fprintf(f, "      \"transactions_per_access\": %.4f,\n",
               transPerAccess(st));
  std::fprintf(f, "      \"barriers\": %llu,\n",
               static_cast<unsigned long long>(st.barriers));
  std::fprintf(f, "      \"bench_reps\": %d,\n", row.reps);
  std::fprintf(f, "      \"bench_seconds\": %.4f,\n", row.seconds);
  std::fprintf(f, "      \"instr_per_sec\": %.0f\n", row.instr_per_sec);
  std::fprintf(f, "    }%s\n", last ? "" : ",");
}

// Hand computed micro patterns, re measured here so the JSON holds real runs.
struct HandCase {
  const char *name;
  uint64_t lds_accesses, lds_conflicts, global_accesses, global_transactions;
};

Stats runPattern(bool lds, int shift, int wave_width, bool broadcast) {
  KB b("pattern");
  LaunchConfig cfg;
  cfg.wave_width = wave_width;
  cfg.workgroup_size = wave_width;
  cfg.num_workgroups = 1;
  GlobalMemory mem(1u << 20);
  uint32_t base = mem.alloc(64 * 1024);
  if (broadcast) {
    if (lds)
      b.lds_store(Imm(128), V(KB::kVLane));
    else
      b.global_load(4, S(KB::kSArg0));
  } else {
    b.vshl_b32(3, V(KB::kVLane), Imm(static_cast<uint32_t>(shift)));
    if (lds) {
      b.lds_store(V(3), V(KB::kVLane));
    } else {
      b.vadd_i32(3, V(3), S(KB::kSArg0));
      b.global_load(4, V(3));
    }
  }
  b.end();
  Program p = b.finish();
  return runProgram(p, cfg, mem, {base});
}

} // namespace

int main() {
  const uint32_t seed = 42;
  std::vector<BenchRow> w32, w64;
  for (const std::string &name : kernelNames()) {
    std::printf("bench %s wave32...\n", name.c_str());
    w32.push_back(benchKernel(name, 32, seed));
    std::printf("bench %s wave64...\n", name.c_str());
    w64.push_back(benchKernel(name, 64, seed));
  }

  // Determinism: identical seeds must give identical outputs and stats.
  KernelRunResult d1 = runKernel("matmul_tiled_f32", 32, 7);
  KernelRunResult d2 = runKernel("matmul_tiled_f32", 32, 7);
  bool deterministic = d1.output_hash == d2.output_hash && d1.stats == d2.stats;

  FILE *f = std::fopen("results/results.json", "w");
  if (!f) {
    std::printf("cannot open results/results.json\n");
    return 1;
  }
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"machine\": \"Apple Silicon arm64, single thread, "
                  "clang++ -O2, stats model not timing model\",\n");
  std::fprintf(f, "  \"seed\": %u,\n", seed);
  std::fprintf(f, "  \"wave32\": [\n");
  for (size_t i = 0; i < w32.size(); i++)
    printRowJson(f, w32[i], i + 1 == w32.size());
  std::fprintf(f, "  ],\n");
  std::fprintf(f, "  \"wave64\": [\n");
  for (size_t i = 0; i < w64.size(); i++)
    printRowJson(f, w64[i], i + 1 == w64.size());
  std::fprintf(f, "  ],\n");

  std::fprintf(f, "  \"wave32_vs_wave64_outputs_identical\": [\n");
  for (size_t i = 0; i < w32.size(); i++) {
    bool same = w32[i].res.output_hash == w64[i].res.output_hash;
    std::fprintf(f, "    {\"kernel\": \"%s\", \"identical\": %s}%s\n",
                 w32[i].res.name.c_str(), same ? "true" : "false",
                 i + 1 == w32.size() ? "" : ",");
  }
  std::fprintf(f, "  ],\n");

  std::fprintf(f, "  \"hand_computed_cases\": [\n");
  struct Row {
    const char *name;
    Stats st;
    uint64_t expect_conf_or_trans;
  };
  std::vector<Row> hand = {
      {"lds_stride_1_wave32", runPattern(true, 2, 32, false), 0},
      {"lds_stride_2_wave32", runPattern(true, 3, 32, false), 1},
      {"lds_stride_32_wave32", runPattern(true, 7, 32, false), 31},
      {"lds_broadcast_wave32", runPattern(true, 0, 32, true), 0},
      {"lds_stride_32_wave64", runPattern(true, 7, 64, false), 62},
      {"global_contiguous_wave32", runPattern(false, 2, 32, false), 2},
      {"global_stride_8B_wave32", runPattern(false, 3, 32, false), 4},
      {"global_stride_64B_wave32", runPattern(false, 6, 32, false), 32},
      {"global_broadcast_wave32", runPattern(false, 0, 32, true), 1},
  };
  for (size_t i = 0; i < hand.size(); i++) {
    const Row &r = hand[i];
    bool isLds = r.st.lds_accesses > 0;
    uint64_t measured =
        isLds ? r.st.lds_conflict_cycles : r.st.global_transactions;
    std::fprintf(f,
                 "    {\"case\": \"%s\", \"measured\": %llu, \"expected\": "
                 "%llu, \"match\": %s}%s\n",
                 r.name, static_cast<unsigned long long>(measured),
                 static_cast<unsigned long long>(r.expect_conf_or_trans),
                 measured == r.expect_conf_or_trans ? "true" : "false",
                 i + 1 == hand.size() ? "" : ",");
  }
  std::fprintf(f, "  ],\n");
  std::fprintf(f, "  \"deterministic_same_seed\": %s\n",
               deterministic ? "true" : "false");
  std::fprintf(f, "}\n");
  std::fclose(f);

  std::printf("\nwrote results/results.json\n");
  for (const BenchRow &r : w32)
    std::printf("%-24s wave32  instr=%9llu  div=%6.2f%%  ldsconf=%7llu  "
                "trans/access=%6.2f  %.2fM instr/s  oracle=%s\n",
                r.res.name.c_str(),
                static_cast<unsigned long long>(r.res.stats.total_instructions),
                divergencePct(r.res.stats),
                static_cast<unsigned long long>(r.res.stats.lds_conflict_cycles),
                transPerAccess(r.res.stats), r.instr_per_sec / 1e6,
                r.res.oracle_match ? "pass" : "FAIL");
  return 0;
}
