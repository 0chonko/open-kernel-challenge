# Open Kernel Challenge · H200 Inference Optimization

CUDA kernel experiments for LLM inference on a single NVIDIA H200. The Open Kernel Challenge supplied the task specifications, reference implementations, and correctness harness. SGLang is the performance baseline. I implemented the CUDA and cuBLASLt submission paths, ran the kernel experiments, and analyzed bottlenecks across input shapes.

| Experiment | Geometric mean relative to SGLang | Implementation | Per-shape evidence |
| --- | --- | --- | --- |
| Dense BF16 projection | 1.43× | [Measured submission](https://github.com/0chonko/open-kernel-challenge/blob/a132ea6/muse_bf16_proj/submission.py) · [Task](muse_bf16_proj/TASK.md) · [Checks](muse_bf16_proj/checks.py) | [Per-case heatmap](https://0chonko.github.io/assets/blog/open-kernel-challenge/bf16-case-heatmap.svg) |
| Attention decode | 1.15× | [Measured submission](https://github.com/0chonko/open-kernel-challenge/blob/2d001cc/muse_attention_decode/submission.py) · [Task](muse_attention_decode/TASK.md) · [Checks](muse_attention_decode/checks.py) | [Per-case heatmap](https://0chonko.github.io/assets/blog/open-kernel-challenge/attn-case-heatmap.svg) |

The task-provided checks and reference implementations define the correctness comparisons. The speed ratios summarize the tested kernel cases on one H200. The [experiment write-up](https://0chonko.github.io/blog/open-kernel-challenge-side-channel/) has the per-shape results, baseline profiles, and implementation tradeoffs.

## Exploratory work

- [Decode layer](muse_decode_layer/), [prefill layer](muse_prefill_layer/), and [W4A16 MoE](muse_moe_w4a16/) hold separate challenge experiments.
- [Serving](serving/) holds end-to-end serving experiments and [papers/index.txt](papers/index.txt) lists the papers consulted.
