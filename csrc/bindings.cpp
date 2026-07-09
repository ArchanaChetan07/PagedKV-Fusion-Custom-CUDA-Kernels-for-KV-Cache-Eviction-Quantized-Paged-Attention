// PagedKV-Fusion — Python bindings for the CUDA kernels.
#include <torch/extension.h>

torch::Tensor eviction_scores_cuda(
    torch::Tensor recency, torch::Tensor frequency, torch::Tensor attn_weights,
    torch::Tensor valid_mask, double w_recency, double w_frequency,
    double w_attention);

torch::Tensor quant_paged_attention_cuda(
    torch::Tensor q, torch::Tensor k_cache_q, torch::Tensor k_scale,
    torch::Tensor v_cache_q, torch::Tensor v_scale, torch::Tensor block_tables,
    torch::Tensor seq_lens, double sm_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PagedKV-Fusion custom CUDA ops";
  m.def("eviction_scores", &eviction_scores_cuda,
        "Fused KV-cache eviction scoring (CUDA)",
        py::arg("recency"), py::arg("frequency"), py::arg("attn_weights"),
        py::arg("valid_mask"), py::arg("w_recency") = 0.4,
        py::arg("w_frequency") = 0.3, py::arg("w_attention") = 0.3);
  m.def("quant_paged_attention", &quant_paged_attention_cuda,
        "INT8 quantized paged attention, decode step (CUDA)",
        py::arg("q"), py::arg("k_cache_q"), py::arg("k_scale"),
        py::arg("v_cache_q"), py::arg("v_scale"), py::arg("block_tables"),
        py::arg("seq_lens"), py::arg("sm_scale"));
}
