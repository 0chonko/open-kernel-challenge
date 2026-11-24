# Open Kernel Challenge · H200 inference optimization

CUDA kernel experiments for LLM inference on a single NVIDIA H200. I used custom CUDA paths and cuBLASLt to investigate bottlenecks that vary with input shape.

| Experiment | Geometric mean relative to SGLang | Implementation |
| --- | --- | --- |
| Dense BF16 projection | 1.43× | [Submission](muse_bf16_proj/submission.py) · [Task](muse_bf16_proj/TASK.md) · [Checks](muse_bf16_proj/checks.py) |
| Attention decode | 1.15× | [Submission](muse_attention_decode/submission.py) · [Task](muse_attention_decode/TASK.md) · [Checks](muse_attention_decode/checks.py) |

The task-provided checks and reference implementations define the correctness comparisons. The speed ratios summarize the tested kernel cases on one H200. The [experiment write-up](https://0chonko.github.io/blog/open-kernel-challenge-side-channel/) has the per-shape results, baseline profiles, and implementation tradeoffs.
