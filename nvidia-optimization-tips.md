# NVIDIA H200 (Compute 9.0a) Deep Kernel Optimization Reference

An advanced reference sheet mapping Hopper-specific hardware constraints to low-level CUDA, PTX, and assembly-level instruction sets.

---

## 1. Hardware Architecture Constraints & Topography

### Streaming Multiprocessor (SM) Topography
* **Active SM Count:** 132 SMs per GPU.
* **Warp Schedulers per SM:** 4 independent warp schedulers.
* **Max Threads per SM:** 2,048 threads (64 warps).
* **Max Thread Blocks per SM:** 32 blocks.

### Memory Subsystem Capacity & Allocation
* **Global Memory (HBM3e):** 141 GB @ 4.8 TB/s peak bandwidth.
* **L2 Cache:** 50 MB dedicated pool with programmatically configurable residency/persistence profiles.
* **Unified L1 / Shared Memory Pool:** 256 KB total SRAM space per SM.
* **Max Configurable Shared Memory:** 228 KB per SM (via `cudaFuncSetAttribute` using `PreferredSharedMemoryCarveout`).
* **Max Shared Memory per Block:** 227 KB (CUDA hard-reserves 1 KB per thread block).
* **Static Allocation Limit:** 48 KB. Dynamic opt-in required via `cudaFuncSetAttribute` to exceed this limit.

---

## 2. Low-Level Execution Frameworks (NVIDIA Mandated)

### A. Warp Specialization Architecture
To maximize Tensor Core efficiency, NVIDIA recommends splitting your thread blocks into distinct, persistent asynchronous hardware roles:
* **Producer Warps:** Dedicated entirely to data marshaling. They calculate spatial indexing and issue asynchronous copy commands directly from global memory to shared memory.
* **Consumer Warps:** Dedicated entirely to arithmetic compute. They wait on synchronization primitives and feed the high-throughput Matrix Multiply-Accumulate (MMA) pipes continuously.

### B. Asynchronous WGMMA (Warp Group Matrix Multiply-Accumulate)
* **Instruction Target:** Avoid basic, synchronous `mma.sync` loops. Hopper introduces **`wgmma.mma_async`** instructions.
* **Execution Mechanics:** Runs natively across an entire unified *Warp Group* (typically 4 or 8 warps working in parallel coherence). 
* **Register Efficiency:** Inputs are read directly out of **Shared Memory** instead of requiring a prior intermediate load pass into registers (`LDG` -> `LDS` -> `Register` is completely bypassed), maximizing register availability for output accumulation.

### C. TMA (Tensor Memory Accelerator) & Co-Design Hardware
* **Hardware Unit:** A dedicated hardware coprocessor that transfers 1D up to 5D multidimensional tensor blocks directly between HBM3e and Shared Memory asynchronously.
* **Instruction Pass:** Uses specialized descriptor definitions. Replaces the legacy manual `cp.async` pointer loop patterns entirely.
* **Synchronization Object:** Track using **Asynchronous Transaction Barriers**. The producer warp initializes a barrier count corresponding to the exact byte-payload of the TMA payload. The hardware decrements this transaction barrier automatically as raw bytes arrive in SRAM, releasing consumer warps instantly without polling.

### D. Thread Block Clusters & DSMEM
* **Cluster Dimension:** Up to 8 blocks can be contextually grouped into an execution `cluster` via `__cluster_dims__(x,y,z)`.
* **Distributed Shared Memory (DSMEM):** Threads can directly address the shared memory space of a separate SM within the same cluster via the local network-on-chip (NoC) using ordinary pointer maps.
* **Occupancy Note:** Always query and compute metrics using `cudaOccupancyMaxActiveClusters` to determine precise occupancy when clustering is active.

---

## 3. Register Breakpoints & Occupancy Limits

Register allocation occurs strictly at a **256-register granularity per warp**. Use the register-per-thread limits below to target required SM utilization rates.

| Target Occupancy | Max Warps / SM | Max Threads / SM | Hard Register Boundary / Thread |
| :--- | :--- | :--- | :--- |
| **100% (Peak)** | 64 Warps | 2,048 Threads | **≤ 32 Registers** |
| **78%** | 50 Warps | 1,600 Threads | **≤ 40 Registers** |
| **50%** | 32 Warps | 1,024 Threads | **≤ 64 Registers** |
| **Minimum** | 8 Warps | 256 Threads | **≤ 255 Registers** (Hardware Cap) |

*Enforcement:* Decorate entry methods with `__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)` to prevent unexpected register spills into slow local memory slots during optimization.

---

## 4. Hardware Arithmetic Core Throughput

Theoretical native structural operations per clock cycle per single physical SM:

| Instruction Class / Target Precision | Native Hardware Operations per SM per Clock Cycle |
| :--- | :--- |
| **FP32 Scalar** (Improved Hopper pipeline) | 128 ops / clock |
| **FP64 Scalar** (Scientific Compute) | 64 ops / clock |
| **INT32** (Index Evaluation & Logical Math) | 64 ops / clock |
| **FP16 Tensor Core** (WGMMA / WMMA) | 256 ops / clock |
| **BF16 / FP8 Tensor Core** (WGMMA Core Target) | 512 ops / clock |
| **DPX Instructions** (Dynamic Programming Acceleration) | Explicit hardware acceleration for sequence alignment (`__viaddmin`, etc.) |

---

## 5. Memory Layout Optimization

### Global Memory Coalescing
* Ensure all global pointers address `32-byte`, `64-byte`, or `128-byte` continuous hardware boundaries. 
* Avoid non-unit stride tracking loops across a warp to guarantee single-cycle transactional memory fetches.

### Shared Memory Bank Protection
* Shared memory is divided into **32 independent banks** spanning a **32-bit (4-byte) alignment grid**.
* Simultaneous address access by threads within the same warp to distinct offsets falling inside the identical 32-bit bank triggers serialization.
* *Fix:* Introduce an array padding index to structural buffers (e.g., changing allocation footprint definitions from `[32][32]` to `[32][33]`) to force uniform address skewing across distinct physical banks.

