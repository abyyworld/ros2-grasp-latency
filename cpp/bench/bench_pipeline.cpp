// In-process benchmark for the C++ core. The whole frame store is read into
// memory before the loop, both output files are written after it, and nothing
// between the two allocates, formats a string or touches a file: the point of
// the exercise is to time the pipeline, not the harness around it.
//
//   bench_pipeline --dataset data/table_640x480
//                  --out-timing results/cpp.timing.jsonl
//                  --out-output results/cpp.output.jsonl
//                  --warmup 100 --frames 2000

#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

#include "grasp_core/config.hpp"
#include "grasp_core/json.hpp"
#include "grasp_core/pipeline.hpp"

namespace
{

struct Options
{
  std::string dataset;
  std::string out_timing;
  std::string out_output;
  std::string config_path{"assets/pipeline_config.json"};
  std::string chain_path{"assets/franka/panda_chain.json"};
  std::string ransac_path{"assets/ransac_uniform.bin"};
  std::string impl{"cpp"};
  int warmup{-1};
  int frames{-1};
};

[[noreturn]] void usage(const char * program, const std::string & why)
{
  std::cerr << why << "\n"
            << "usage: " << program << " --dataset DIR --out-timing FILE --out-output FILE\n"
            << "       [--warmup N] [--frames N] [--impl NAME]\n"
            << "       [--config FILE] [--chain FILE] [--ransac-table FILE]\n";
  std::exit(2);
}

Options parse_options(int argc, char ** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_value = (i + 1) < argc;
    auto value = [&]() -> std::string {
        if (!has_value) {
          usage(argv[0], "missing value for " + flag);
        }
        return argv[++i];
      };
    if (flag == "--dataset") {
      o.dataset = value();
    } else if (flag == "--out-timing") {
      o.out_timing = value();
    } else if (flag == "--out-output") {
      o.out_output = value();
    } else if (flag == "--config") {
      o.config_path = value();
    } else if (flag == "--chain") {
      o.chain_path = value();
    } else if (flag == "--ransac-table") {
      o.ransac_path = value();
    } else if (flag == "--impl") {
      o.impl = value();
    } else if (flag == "--warmup") {
      o.warmup = std::atoi(value().c_str());
    } else if (flag == "--frames") {
      o.frames = std::atoi(value().c_str());
    } else {
      usage(argv[0], "unknown flag " + flag);
    }
  }
  if (o.dataset.empty() || o.out_timing.empty() || o.out_output.empty()) {
    usage(argv[0], "--dataset, --out-timing and --out-output are all required");
  }
  return o;
}

struct FrameStore
{
  std::string name;
  int width{0};
  int height{0};
  int count{0};
  std::vector<std::uint16_t> depth;   // count * width * height
  std::vector<std::uint8_t> rgb;      // count * width * height * 3
};

void read_exact(const std::string & path, void * dst, std::size_t bytes)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  in.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
  if (static_cast<std::size_t>(in.gcount()) != bytes) {
    throw std::runtime_error(path + " is shorter than the manifest says");
  }
}

FrameStore load_store(const std::string & directory)
{
  using grasp_core::json::Value;
  const Value manifest = grasp_core::json::parse_file(directory + "/manifest.json");

  FrameStore store;
  store.name = manifest["name"].string();
  store.width = manifest["width"].integer();
  store.height = manifest["height"].integer();
  const auto & frames = manifest["frames"].array();
  store.count = static_cast<int>(frames.size());
  if (store.count < 1) {
    throw std::runtime_error(directory + " lists no frames");
  }

  const std::size_t pixels = static_cast<std::size_t>(store.width) *
    static_cast<std::size_t>(store.height);
  store.depth.resize(pixels * static_cast<std::size_t>(store.count));
  store.rgb.resize(pixels * 3u * static_cast<std::size_t>(store.count));

  for (int i = 0; i < store.count; ++i) {
    const auto & entry = frames[static_cast<std::size_t>(i)];
    read_exact(
      directory + "/" + entry["depth"].string(),
      store.depth.data() + pixels * static_cast<std::size_t>(i),
      pixels * sizeof(std::uint16_t));
    read_exact(
      directory + "/" + entry["rgb"].string(),
      store.rgb.data() + pixels * 3u * static_cast<std::size_t>(i),
      pixels * 3u);
  }
  return store;
}

struct TimingRecord
{
  int frame{0};
  int seq{0};
  grasp_core::StageNs stage;
  std::uint64_t total_ns{0};
  int points{0};
  int cluster_points{0};
  int ik_iterations{0};
  bool plane_found{false};
  bool graspable{false};
  bool converged{false};
};

struct OutputRecord
{
  bool seen{false};
  bool plane_found{false};
  std::array<double, 4> plane{};
  int cluster_points{0};
  bool graspable{false};
  double width{0.0};
  std::array<double, 16> tcp{};
  bool converged{false};
  int iterations{0};
  std::array<double, grasp_core::kDof> q{};
  double duration_s{0.0};
  // waypoints * (3 * kDof + 1), in the order docs/FORMATS.md fixes: one
  // waypoint's positions, velocities, accelerations and time, then the next.
  std::vector<double> waypoint_block;
};

// %.17g so a double survives a round trip through JSON unchanged.
void append_double(std::string & out, double v)
{
  char buffer[48];
  std::snprintf(buffer, sizeof(buffer), "%.17g", v);
  out += buffer;
}

void append_doubles(std::string & out, const double * values, std::size_t n)
{
  out += '[';
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) {
      out += ',';
    }
    append_double(out, values[i]);
  }
  out += ']';
}

void append_int(std::string & out, long long v)
{
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%lld", v);
  out += buffer;
}

void append_bool(std::string & out, bool v)
{
  out += v ? "true" : "false";
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Options opt = parse_options(argc, argv);
    const FrameStore store = load_store(opt.dataset);

    grasp_core::Pipeline pipeline(opt.config_path, opt.chain_path, opt.ransac_path);
    const grasp_core::Config & cfg = pipeline.config();

    const int warmup = opt.warmup >= 0 ? opt.warmup : cfg.benchmark.warmup_frames;
    const int measured = opt.frames > 0 ? opt.frames : cfg.benchmark.measured_frames;
    const std::size_t pixels = static_cast<std::size_t>(store.width) *
      static_cast<std::size_t>(store.height);
    const std::size_t block_size =
      static_cast<std::size_t>(cfg.trajectory.waypoints) * (3u * grasp_core::kDof + 1u);

    // Everything the loop will touch is allocated here, including the
    // pipeline's own scratch, so the first measured frame is no different from
    // the last one.
    pipeline.reserve(store.width, store.height);
    std::vector<TimingRecord> timings(static_cast<std::size_t>(measured));
    std::vector<OutputRecord> outputs(static_cast<std::size_t>(store.count));
    for (auto & record : outputs) {
      record.waypoint_block.resize(block_size);
    }

    const int total_frames = warmup + measured;
    for (int i = 0; i < total_frames; ++i) {
      const int frame = i % store.count;
      const grasp_core::Result & result = pipeline.run(
        store.depth.data() + pixels * static_cast<std::size_t>(frame),
        store.rgb.data() + pixels * 3u * static_cast<std::size_t>(frame),
        store.width, store.height);
      if (i < warmup) {
        continue;
      }

      const int seq = i - warmup;
      TimingRecord & t = timings[static_cast<std::size_t>(seq)];
      t.frame = frame;
      t.seq = seq;
      t.stage = result.stage_ns;
      t.total_ns = result.total_ns;
      t.points = result.points;
      t.cluster_points = result.cluster_points;
      t.ik_iterations = result.iterations;
      t.plane_found = result.plane_found;
      t.graspable = result.graspable;
      t.converged = result.converged;

      // The pipeline is deterministic, so a frame seen twice produces the same
      // outputs; recording it once keeps the equivalence file one line per
      // frame of the store rather than one line per measured sample.
      OutputRecord & o = outputs[static_cast<std::size_t>(frame)];
      if (!o.seen) {
        o.seen = true;
        o.plane_found = result.plane_found;
        o.plane = result.plane;
        o.cluster_points = result.cluster_points;
        o.graspable = result.graspable;
        o.width = result.width;
        o.tcp = result.tcp;
        o.converged = result.converged;
        o.iterations = result.iterations;
        o.q = result.q;
        o.duration_s = result.duration_s;
        std::size_t at = 0;
        for (const auto & w : result.trajectory) {
          for (int j = 0; j < grasp_core::kDof; ++j) {
            o.waypoint_block[at++] = w.position[static_cast<std::size_t>(j)];
          }
          for (int j = 0; j < grasp_core::kDof; ++j) {
            o.waypoint_block[at++] = w.velocity[static_cast<std::size_t>(j)];
          }
          for (int j = 0; j < grasp_core::kDof; ++j) {
            o.waypoint_block[at++] = w.acceleration[static_cast<std::size_t>(j)];
          }
          o.waypoint_block[at++] = w.time_from_start;
        }
      }
    }

    const long long overhead_ns = static_cast<long long>(std::llround(
        pipeline.timer_overhead_ns()));

    std::string text;
    text.reserve(static_cast<std::size_t>(measured) * 400u);
    for (const TimingRecord & t : timings) {
      text += "{\"impl\":\"" + opt.impl + "\",\"dataset\":\"" + store.name + "\",\"frame\":";
      append_int(text, t.frame);
      text += ",\"seq\":";
      append_int(text, t.seq);
      text += ",\"stage_ns\":{\"decode\":";
      append_int(text, static_cast<long long>(t.stage.decode));
      text += ",\"deproject\":";
      append_int(text, static_cast<long long>(t.stage.deproject));
      text += ",\"transform_crop\":";
      append_int(text, static_cast<long long>(t.stage.transform_crop));
      text += ",\"plane\":";
      append_int(text, static_cast<long long>(t.stage.plane));
      text += ",\"cluster\":";
      append_int(text, static_cast<long long>(t.stage.cluster));
      text += ",\"grasp\":";
      append_int(text, static_cast<long long>(t.stage.grasp));
      text += ",\"ik\":";
      append_int(text, static_cast<long long>(t.stage.ik));
      text += ",\"traj\":";
      append_int(text, static_cast<long long>(t.stage.traj));
      text += "},\"total_ns\":";
      append_int(text, static_cast<long long>(t.total_ns));
      text += ",\"points\":";
      append_int(text, t.points);
      text += ",\"cluster_points\":";
      append_int(text, t.cluster_points);
      text += ",\"ik_iterations\":";
      append_int(text, t.ik_iterations);
      text += ",\"plane_found\":";
      append_bool(text, t.plane_found);
      text += ",\"graspable\":";
      append_bool(text, t.graspable);
      text += ",\"converged\":";
      append_bool(text, t.converged);
      text += ",\"timer_overhead_ns\":";
      append_int(text, overhead_ns);
      text += "}\n";
    }
    {
      std::ofstream out(opt.out_timing, std::ios::binary);
      if (!out) {
        throw std::runtime_error("cannot write " + opt.out_timing);
      }
      out << text;
    }

    text.clear();
    // Roughly 24 characters per %.17g double, plus the fixed fields. One
    // reserve rather than a dozen reallocations while the block is appended.
    text.reserve(static_cast<std::size_t>(store.count) * (24u * (block_size + 32u)));
    for (int frame = 0; frame < store.count; ++frame) {
      const OutputRecord & o = outputs[static_cast<std::size_t>(frame)];
      if (!o.seen) {
        continue;
      }
      text += "{\"impl\":\"" + opt.impl + "\",\"frame\":";
      append_int(text, frame);
      text += ",\"plane_found\":";
      append_bool(text, o.plane_found);
      text += ",\"plane\":";
      append_doubles(text, o.plane.data(), o.plane.size());
      text += ",\"cluster_points\":";
      append_int(text, o.cluster_points);
      text += ",\"graspable\":";
      append_bool(text, o.graspable);
      text += ",\"width\":";
      append_double(text, o.width);
      text += ",\"tcp\":";
      append_doubles(text, o.tcp.data(), o.tcp.size());
      text += ",\"converged\":";
      append_bool(text, o.converged);
      text += ",\"iterations\":";
      append_int(text, o.iterations);
      text += ",\"q\":";
      append_doubles(text, o.q.data(), o.q.size());
      text += ",\"duration_s\":";
      append_double(text, o.duration_s);
      text += ",\"trajectory\":";
      append_doubles(text, o.waypoint_block.data(), o.waypoint_block.size());
      text += "}\n";
    }
    {
      std::ofstream out(opt.out_output, std::ios::binary);
      if (!out) {
        throw std::runtime_error("cannot write " + opt.out_output);
      }
      out << text;
    }

    std::cerr << "dataset=" << store.name << " " << store.width << "x" << store.height
              << " frames=" << store.count << " warmup=" << warmup
              << " measured=" << measured
              << " timer_overhead_ns=" << overhead_ns << "\n";
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "bench_pipeline: " << e.what() << "\n";
    return 1;
  }
}
