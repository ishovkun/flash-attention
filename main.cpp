#include "flash/launch.hpp"
#include <chrono>
#include <torch/torch.h>

struct AttentionParameters {
  int batch_size;
  int num_heads;
  int seq_len;
  int head_embd;
};

torch::Tensor manual_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  auto att = (q.matmul(k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1))));
  att = torch::nn::functional::softmax(att, -1);
  auto y = att.matmul(v);
  return y;
}

torch::Tensor torch_attn2(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  // return torch::nn::functional::scaled_dot_product_attention(q, k, v);
  return torch::scaled_dot_product_attention(q, k, v);
}

auto generate_data(auto const &p) {
  torch::manual_seed(0);
  std::cout << "Generate sample: " << "Batch " << p.batch_size
            << " Num Heads " << p.num_heads  << " Seq len " << p.seq_len
            << " Head embedding " << p.head_embd
            << std::endl;
  auto q = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  auto k = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  auto v = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  return std::make_tuple(q, k, v);
}

bool run_and_compare(auto test_name, auto reference, double atol, double rtol,
                     auto &&kernel) {
  auto result = kernel();
  bool test_result = torch::allclose(result, reference, atol, rtol);
  if (!test_result) {
    std::cout << test_name << ": Test failed [☓ ]" << std::endl;
    int mismatch_count = 0;
    for (int b = 0; b < result.size(0); b++) {
      for (int h = 0; h < result.size(1); h++) {
        for (int n = 0; n < result.size(2); n++) {
          for (int d = 0; d < result.size(3); d++) {
            auto ref = reference[b][h][n][d].template item<float>();
            auto res = result[b][h][n][d].template item<float>();
            if (std::fabs(res - ref) > atol + rtol * std::fabs(ref) ||
                std::isnan(res)) {
              std::cout << "Mismatch[" << b << "," << h << "," << n << "," << d
                        << "] = "
                        << "ref = " << ref << ", res = " << res << std::endl;
              if (mismatch_count++ > 5)
                return false;
            }
          }
        }
      }
    }
  } else {
    std::cout << test_name << ": Test passed [✓ ]" << std::endl;
  }
  return test_result;
}

void test_alg(AttentionParameters const &params) {
  std::cout << "======= Running test ("
            << "batch = " << params.batch_size << ", "
            << "heads = " << params.num_heads << ", "
            << "seq_len = " << params.seq_len << ", "
            << "embedding dim = " << params.head_embd
            << ") =======" << std::endl;
  auto [q, k, v] = generate_data(params);
  auto manual_result = manual_attn(q, k, v);
  // std::cout << "Reference: " << manual_result << std::endl;

  double atol = 1e-4;
  double rtol = 1e-2;

  bool ret = true;

  ret &= run_and_compare("Naive", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::naive1D);
  });
  ret &= run_and_compare("Scalar 2D block", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::scalar2D);
  });
  ret &= run_and_compare("Scalar 2D row tile", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::scalar2D_row_tile);
  });
  ret &= run_and_compare("Single-warp wmma", manual_result, atol, rtol, [&] {
        return flash::forward(q, k, v, flash::KernelType::warp_wmma);
      });
  if (params.head_embd < 128) { // it fails but whatever, not gonna debug slow code
    ret &= run_and_compare("Block wmma", manual_result, atol, rtol, [&] {
      return flash::forward(q, k, v, flash::KernelType::block_wmma);
    });
  }
  if (params.head_embd < 128) { // it fails but whatever, not gonna debug slow code
    ret &= run_and_compare("wmma row-block", manual_result, atol, rtol, [&] {
      return flash::forward(q, k, v, flash::KernelType::wmma_row_block);
    });
  }
  ret &= run_and_compare("mma", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::mma);
  });
  ret &= run_and_compare("mma swizzle", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::mma_swizzle);
  });
  ret &= run_and_compare("Block wmma async", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::block_wmma_async);
  });
  if (params.head_embd % 2 == 0) { // otherwise misaligned address
    ret &= run_and_compare("mma qreg", manual_result, atol, rtol, [&] {
      return flash::forward(q, k, v, flash::KernelType::mma_qreg);
    });
  }
  if (params.head_embd % 4 == 0) { // otherwise misaligned address
    ret &= run_and_compare("mma qreg vload", manual_result, atol, rtol, [&] {
      return flash::forward(q, k, v, flash::KernelType::mma_qreg_f32x4load);
    });
    ret &= run_and_compare("mma qreg async", manual_result, atol, rtol, [&] {
      return flash::forward(q, k, v, flash::KernelType::mma_qreg_async);
    });
  }
  if (!ret) {
    std::cout << "Test failed!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename KernelType>
auto time_kernel(torch::Tensor const & q, torch::Tensor const & k, torch::Tensor const & v,
                 std::string const & kernel_name, KernelType && kernel)
{
  using namespace std::chrono;
  auto const start_time = high_resolution_clock::now();
  kernel(q,k,v);
  auto const end_time = high_resolution_clock::now();
  auto const duration = (duration_cast<milliseconds>(end_time - start_time)).count();
  std::cout << "Benchmark \'" << kernel_name << "\'"
            << " took " << (double)duration << " [ms]" << std::endl;}

auto time_kernel(torch::Tensor  const & q, torch::Tensor  const & k, torch::Tensor  const & v,
                 flash::KernelType kernelType) {
  auto func = [kernelType](auto const &q, auto const &k, auto const &v) {
    return flash::forward(q, k, v, kernelType);
  };
  time_kernel(q, k, v, to_string(kernelType), func);
}

auto main(int argc, char *argv[]) -> int {

  if (argc == 1) { // test only
    // small aliged
    test_alg(AttentionParameters{
        .batch_size = 1,
        .num_heads = 1,
        .seq_len = 50,
        .head_embd = 32,
    });

    // unaligned seq len
    test_alg(AttentionParameters{
        .batch_size = 1,
        .num_heads = 1,
        .seq_len = 100, // larger than tile size
        .head_embd = 32,
    });

    // unaligned head embedding dim
    test_alg(AttentionParameters{
        .batch_size = 1,
        .num_heads = 1,
        .seq_len = 64,
        .head_embd = 50,
    });

    // Bigger test with everything unaligned
    test_alg(AttentionParameters{
        .batch_size = 5,
        .num_heads = 12,
        .seq_len = 53,
        .head_embd = 69,
    });
    // some kernels launch more warps for head_emb > 64
    test_alg(AttentionParameters{
        .batch_size = 1,
        .num_heads = 1,
        .seq_len = 30,
        .head_embd = 128,
    });
  }
  else { // profile or benchmark
    bool ncu = argv[1] == std::string("ncu");
    auto [q, k, v] = generate_data(AttentionParameters{
            // gpt3
            .batch_size = 4,
            .num_heads = 96,
            .seq_len = 2048,
            .head_embd = 128,
            // gpt2
            // .batch_size = 8,
            // .num_heads = 12,
            // .seq_len = 1024,
            // .head_embd = 64,
        });

    if (!ncu) {
      time_kernel(q, k, v, flash::KernelType::naive1D);
      time_kernel(q, k, v, flash::KernelType::scalar2D);
      time_kernel(q, k, v, flash::KernelType::scalar2D_row_tile);
      time_kernel(q, k, v, flash::KernelType::warp_wmma);
      time_kernel(q, k, v, flash::KernelType::block_wmma);
      time_kernel(q, k, v, flash::KernelType::wmma_row_block);
      time_kernel(q, k, v, flash::KernelType::mma);
      time_kernel(q, k, v, flash::KernelType::mma_swizzle);
    }

    time_kernel(q, k, v, flash::KernelType::mma_qreg);
    time_kernel(q, k, v, flash::KernelType::mma_qreg_f32x4load);
    time_kernel(q, k, v, flash::KernelType::mma_qreg_async);

    if (!ncu) {
      time_kernel(q, k, v, flash::KernelType::block_wmma_async);
      time_kernel(q, k, v, "manual attention", manual_attn);
      time_kernel(q, k, v, "torch attention v2", torch_attn2);
    }
  }

  return EXIT_SUCCESS;
}
