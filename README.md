# MCMC Stock Price Forecasting with GPU Acceleration

Monte Carlo Markov Chain simulation engine for stock price forecasting, optimized for NVIDIA CUDA GPUs. Uses regime-switching Markov chains and bootstrap sampling to generate probabilistic price forecasts.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI
python main.py

# Run benchmark (uncomment line 775 in main.py first)
python main.py
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (Tkinter GUI / Direct Function Calls)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Data Acquisition Layer                      │
│  • yfinance API                                             │
│  • Regular trading hours filtering                          │
│  • OHLCV data processing                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 MCMC Simulation Engine                       │
│                  (StockPriceMCMC Class)                      │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │  CPU Path          │         GPU Path         │          │
│  ├──────────────────────────────────────────────┤          │
│  │  NumPy            │         CuPy              │          │
│  │  Serial loops     │    CUDA kernels           │          │
│  │  1K simulations   │    50K simulations        │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Components:                                                 │
│  • Regime detection (quantile-based state inference)        │
│  • Transition matrix builder                                │
│  • Bootstrap sampler (block & simple)                       │
│  • Markov chain simulator                                   │
│  • Geometric Brownian Motion path generator                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Visualization & Analytics Layer                 │
│  • Matplotlib candlestick charts                           │
│  • Percentile bands (5%, 25%, 50%, 75%, 95%)               │
│  • Out-of-sample RMSE calculation                          │
│  • Regime analysis                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## GPU Acceleration Architecture

### Why GPU?

Monte Carlo simulations are **embarrassingly parallel** - each simulation path is independent. This is perfect for GPUs:

| CPU | GPU |
|-----|-----|
| 8-16 cores | 10,000+ CUDA cores |
| Sequential processing | Massive parallelism |
| ~50 GB/s memory bandwidth | ~1 TB/s memory bandwidth |
| 1,000 simulations | 50,000 simulations |

### Parallel Execution Model

```
CPU (Sequential):
Time ─────────────────────────────────────────────>
Sim 1: [████████████████████]
Sim 2:                      [████████████████████]
Sim 3:                                           [████████████████████]
...
Total: N × T time

GPU (Parallel):
Time ─────>
Sim 1:    [██]
Sim 2:    [██]
Sim 3:    [██]
...
Sim 50K:  [██]
Total: T time (all simulations run simultaneously)
```

---

## CUDA Kernel Design

### Custom Markov Chain Kernel

The core GPU optimization is a custom CUDA kernel that simulates all Markov chains in parallel:

```cuda
extern "C" __global__
void markov_chain_step(
    const float* trans_matrix,    // Transition probabilities
    const float* rand_vals,       // Pre-generated random values
    int* states,                  // Output: state sequences
    const int n_sims,
    const int n_steps,
    const int n_states,
    const int init_state
)
```

**Thread Organization:**
- **Grid:** `(n_simulations / 256) + 1` blocks
- **Block:** 256 threads
- **Thread-to-simulation mapping:** 1:1 (each thread simulates one complete chain)

**Memory Access Pattern:**
```
Global Memory Layout:
┌─────────────────────────────────────────────────────┐
│  trans_matrix[n_states × n_states]    (shared read) │
│  rand_vals[n_sims × n_steps]          (coalesced)   │
│  states[n_sims × n_steps]             (coalesced)   │
└─────────────────────────────────────────────────────┘

Each thread:
1. Reads its own rand_vals stripe (coalesced access)
2. Reads trans_matrix (cached in L1/L2)
3. Writes its state sequence (coalesced access)
```

**Algorithm:**
```
For each thread (simulation):
    current_state = init_state
    For each timestep:
        r = random_value[thread_id, timestep]
        cumsum = 0
        For each possible next state:
            cumsum += transition_probability[current_state][next_state]
            if r <= cumsum:
                next_state = state
                break
        states[thread_id, timestep] = next_state
        current_state = next_state
```

**Performance Characteristics:**
- **Divergence:** Minimal (inner loop over states is small, typically 3)
- **Memory:** Coalesced reads/writes
- **Occupancy:** 100% (simple kernel, low register usage)

---

## Data Flow Through the System

### End-to-End GPU Pipeline

```
User Request
     │
     ▼
fetch_stock_data()
     │ (CPU: pandas DataFrame)
     ▼
StockPriceMCMC.__init__()
     │ (CPU: store prices)
     ▼
simulate_future_paths()
     │
     ├─> calculate returns (CPU: NumPy log returns)
     │
     ▼
_run_simulation()
     │
     ├─> _generate_returns()
     │        │
     │        ├─> _sample_markov_returns()
     │        │        │
     │        │        ├─> _infer_regimes()        [CPU: pandas qcut]
     │        │        ├─> _build_transition_matrix() [CPU: NumPy]
     │        │        │
     │        │        ▼
     │        │   ┌─────────────────────────────────────┐
     │        │   │  GPU PIPELINE STARTS HERE           │
     │        │   ├─────────────────────────────────────┤
     │        │   │                                     │
     │        │   │  1. Copy transition matrix to GPU   │
     │        │   │     trans_mat: CPU → GPU            │
     │        │   │                                     │
     │        │   │  2. Generate random values on GPU   │
     │        │   │     xp.random.uniform() [on GPU]    │
     │        │   │                                     │
     │        │   │  3. Launch CUDA kernel              │
     │        │   │     markov_kernel<<<blocks, 256>>>  │
     │        │   │     • 50,000 threads in parallel    │
     │        │   │     • Each thread = 1 chain         │
     │        │   │                                     │
     │        │   │  4. Map states to returns (GPU)     │
     │        │   │     Vectorized indexing on GPU      │
     │        │   │                                     │
     │        │   │  [Returns stay on GPU]              │
     │        │   └─────────────────────────────────────┘
     │        │
     │        └──> returns [CuPy array on GPU]
     │
     ├─> Compute price paths (GPU)
     │     • xp.exp(returns)           [GPU]
     │     • xp.cumprod()               [GPU]
     │     • paths = start × cumprod   [GPU]
     │
     │   [Paths stay on GPU]
     │
     ├─> Transfer paths to CPU (one-time transfer)
     │     paths_cpu = cp.asnumpy(paths_gpu)
     │
     ▼
create_chart()
     │
     ├─> [If GPU available]
     │   • Copy paths back to GPU for percentiles
     │   • cp.percentile() [GPU]
     │   • cp.mean() [GPU]
     │   • Transfer only final statistics to CPU
     │
     ▼
Display results
```

### Memory Transfer Points

The system minimizes CPU↔GPU transfers:

**Transfers TO GPU (once per simulation):**
1. Transition matrix (~36 bytes for 3×3)
2. State return values (~few KB)

**Transfers FROM GPU (once per simulation):**
1. Final paths (50K × 31 × 8 bytes = ~12 MB)
2. OR just percentiles (5 × 31 × 8 bytes = ~1 KB)

**Everything else stays on GPU:**
- Random number generation
- State transitions
- Return sampling
- Path computation
- Statistics (if calculated)

---

## Performance

**Computational Complexity:**
```
CPU: O(n_sims × n_steps) sequentially = ~10 seconds
GPU: O(n_steps) in parallel = ~0.5 seconds
Speedup: 20×
```

**Memory Usage:**
```
GPU Memory: ~37 MB total for 50K simulations
Available (RTX 3060): 12 GB
Utilization: 0.3%
```

---

## License

MIT License
