# NVIDIA H200 Architecture & Kernel Optimization Reference

A high-density reference sheet for GPU performance engineering, kernel optimization, and PTX tuning on the NVIDIA H200 (Hopper Architecture).

---

## 1. Hardware Topography & Latency Map

### Streaming Multiprocessor (SM) Topography
* **Active SM Count:** 132 SMs per GPU.
* **Warp Schedulers per SM:** 4 independent warp schedulers.
* **Max Threads per SM:** 2,048 threads (64 warps).
* **Max Thread Blocks per SM:** 32 blocks.

### Memory & Cache Constraints
* **Global Memory (HBM3e):** 141 GB @ 4.8 TB/s bandwidth.
* **L2 Cache:** 50 MB dedicated pool.
* **Max Shared Memory per SM:** 228 KB (configurable via `cudaFuncSetAttribute`).
* **Registers per SM:** 256 KB (65,536 32-bit registers).

### Hardware Latency Reference (Approximate)
* **Register Access:** ~1 clock cycle.
* **Shared Memory / L1 Cache:** ~20–30 clock cycles.
* **L2 Cache Access:** ~150–200 clock cycles.
* **HBM3e (Global Memory):** ~400–600 clock cycles.

---

## 2. Occupancy Rules & Register Boundaries

To achieve maximum SM occupancy (2,048 threads / 64 warps), select block dimensions and register limits based on the hardware allocation granularity below.

### Register Allocation Metrics
* **Allocation Granularity:** 256 registers per warp.
* **Max Registers per Thread:** 255.

### Occupancy Breakpoints
* **64 Registers/Thread:** Up to 1,024 threads per SM (50% Occupancy limit).
* **40 Registers/Thread:** Up to 1,600 threads per SM (78% Occupancy limit).
* **32 Registers/Thread:** Allows full 2,048 threads per SM (100% Occupancy limit).

*Optimization Tip:* Use `__global__ __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` in your CUDA source to force compiler constraints and prevent local memory spilling.

---

## 3. Memory Alignment & Bank Conflicts

### Global Memory Coalescing
* **Transaction Size:** 32, 64, or 128 bytes.
* **Alignment Requirement:** Base pointers must be aligned to 32-byte or 128-byte boundaries.
* **Warp Access Pattern:** Threads in a single warp must access a continuous block of memory to trigger a single combined hardware memory transaction.

### Shared Memory Optimization
* **Bank Count:** 32 independent banks.
* **Bank Width:** 32 bits (4 bytes) per clock cycle.
* **Conflict Condition:** Two or more threads in a warp accessing different addresses within the same 32-bit bank simultaneously results in serialization.
* **Mitigation Strategy:** Pad structural arrays by an extra element (e.g., `__shared__ float shared_data[32][33]`) to offset sequential column indexes into distinct physical banks.

### Hopper-Specific: Thread Block Clusters
* **Cluster Scale:** Group up to 8 cooperative thread blocks.
* **Distributed Shared Memory (DSM):** Allows blocks within the same cluster to read/write directly to neighboring SM shared memory spaces over the NVLink Network On Chip (NoC) without passing through global HBM3e.

---

## 4. Math Core Compute Throughput

Theoretical operations per clock cycle per single SM:

| Instruction Variant / Data Type | Operations per SM per Clock |
| :--- | :--- |
| **FP32 (Single Precision Scalar)** | 128 ops/clock |
| **FP64 (Double Precision Scalar)** | 64 ops/clock |
| **INT32 (Integer Arithmetic)** | 64 ops/clock |
| **FP16 Tensor Core (MMA)** | 256 ops/clock |
| **BF16 / FP8 Tensor Core (MMA)** | 512 ops/clock |

*Optimization Tip:* Prefer FP8 or FP16 tensor core matrix calculations (`wmma` or `mma.sync` instruction sets) rather than individual loop-unrolled element arithmetic to maximize compute density.

---

## 5. Performance Profiling Triage

Use **Nsight Compute** to pinpoint bottleneck types using the execution stall profiles detailed below:

### Stall: `Stall Memory Throttle`
* **Root Cause:** The pipeline is completely saturated with active global memory transactions.
* **Action:** Review cache utilization. Implement `cooperative_groups` or shared memory buffering to reuse loaded registers and elements.

### Stall: `Stall MIO / Pipe Throttle`
* **Root Cause:** Structural conflict inside the execution pipeline (e.g., executing too many Special Function Unit instructions like `sin`, `cos`, or `log` in consecutive sequences).
* **Action:** Distribute non-standard arithmetic instructions evenly or use lower-precision intrinsic math implementations (e.g., `__sinf()`).

### Stall: `Stall Memory Dependency`
* **Root Cause:** Next instruction requires data that has not returned from high-latency global memory yet.
* **Action:** Increase independent instructions between memory loads, or leverage asynchronous global-to-shared memory operations (`cp.async`) to bypass intermediate register pools.

