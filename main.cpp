#include <torch/torch.h>
#include "flash/launch.hpp"

// torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
// torch::Tensor forward_opt(torch::Tensor q, torch::Tensor k, torch::Tensor v);

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", torch::wrap_pybind_function(forward), "forward");
//     m.def("forward_opt", torch::wrap_pybind_function(forward_opt), "forward_opt");
// }

struct AttentionParameters {
  int batch_size;
  int num_heads;
  int seq_len;
  int head_embd;
};

auto manual_attn(auto q, auto k, auto v) {
  auto att = (q.matmul( k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1))));
  att = torch::nn::functional::softmax(att, -1);
  auto y = att.matmul(v);
  return y;
}

auto generate_data(auto const & p) {
  torch::manual_seed(0);
  auto q = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  auto k = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  auto v = torch::randn({p.batch_size, p.num_heads, p.seq_len, p.head_embd}).cuda();
  return std::make_tuple(q, k, v);
}

bool run_and_compare(auto test_name, auto reference, double atol, double rtol, auto && kernel) {
  auto result = kernel();
  bool test_result = torch::allclose(result, reference, atol, rtol);
  // std::cout << "Reference: " << reference << std::endl;
  // std::cout << "Result: " << result << std::endl;
  if (!test_result) {
    std::cout << test_name << ": Test failed [☓ ]" << std::endl;
    // std::cout << "Reference: " << reference << std::endl;
    // std::cout << "Result: " << result << std::endl;
    int mismatch_count = 0;
    for (int b = 0; b < result.size(0); b++) {
      for (int h = 0; h < result.size(1); h++) {
        for (int n = 0; n < result.size(2); n++) {
          for (int d = 0; d < result.size(3); d++) {
            auto ref = reference[b][h][n][d].template item<float>();
            auto res = result[b][h][n][d].template item<float>();
            if (std::fabs(res - ref) > atol + rtol * std::fabs(ref) || std::isnan(res)) {
              std::cout << "Mismatch[" << b << "," << h << "," << n << "," << d << "] = "
                        << "ref = " << ref << ", res = " << res << std::endl;
              if (mismatch_count++ > 10) return false;
            }
          }
        }
      }
    }
  }
  else {
    std::cout << test_name << ": Test passed [✓ ]" << std::endl;
  }
  return test_result;
}

auto main() -> int {

  // AttentionParameters params{
  //   .batch_size = 1,
  //   .num_heads = 16,
  //   .seq_len = 32,
  //   .head_embd = 32,
  // };

  /* Correctness tests */
  AttentionParameters params{
    .batch_size = 1,
    .num_heads = 20,
    .seq_len = 30,
    .head_embd = 60,
  };

  auto [q, k, v] = generate_data(params);
  auto manual_result = manual_attn(q, k, v);
  // std::cout << "Reference: " << manual_result << std::endl;

  double atol = 1e-4;
  double rtol = 1e-2;

  run_and_compare("Flash naive", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::naive1D);
  });
  run_and_compare("Flash 2D", manual_result, atol, rtol, [&] {
    return flash::forward(q, k, v, flash::KernelType::scalar2D);
  });

  return EXIT_SUCCESS;
}
