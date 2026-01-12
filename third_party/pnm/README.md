# TritonPNM Backend

PNM (Processing Near Memory) backend for Triton.

## Overview

This backend enables Triton to compile kernels for PNM accelerators. PNM devices are specialized hardware that performs computation near memory, reducing data movement overhead for memory-bound workloads.

## Directory Structure

```
pnm/
├── backend/                    # Python backend implementation
│   ├── __init__.py
│   ├── compiler.py            # PNMBackend class
│   ├── driver.py              # PNMDriver class
│   └── name.conf              # Backend name for registration
├── include/
│   └── Dialect/
│       └── TritonPNM/
│           └── IR/
│               ├── TritonPNMDialect.td    # Dialect definition
│               ├── TritonPNMOps.td        # Operations definition
│               ├── TritonPNMTypes.td      # Types definition
│               └── TritonPNMAttrDefs.td   # Attributes definition
├── lib/
│   └── Dialect/
│       └── TritonPNM/
│           ├── IR/                        # Dialect C++ implementation
│           └── Transforms/                # Optimization passes
├── language/                   # PNM-specific language extensions
├── tools/                      # Development tools
├── test/                       # Test suite
├── CMakeLists.txt             # Build configuration
└── README.md                  # This file
```

## Compilation Pipeline

Unlike GPU backends (NVIDIA/AMD) that go through TritonGPU IR, PNM backend converts directly from Triton IR to TritonPNM IR. This is because TritonGPU concepts (warps, CTAs, shared memory) don't apply to PNM's compute unit and local memory architecture.

```
                         GPU Backend (NVIDIA/AMD)
                         ┌─────────────────────────────────────┐
Python @triton.jit ─────►│ TTIR ──► TTGIR ──► LLVM ──► PTX/GCN│
                         └─────────────────────────────────────┘

                         PNM Backend (Direct lowering)
                         ┌─────────────────────────────────────┐
Python @triton.jit ─────►│ TTIR ──► TTPNM ──► PNM ASM ──► BIN │
                         └─────────────────────────────────────┘
```

### Detailed PNM Pipeline

```
Python @triton.jit
       ↓
   Triton IR (ttir)
       ↓ make_ttir()
   Optimized TTIR
       ↓ make_pnmir()        ← Direct lowering (no TritonGPU)
   PNM IR (ttpnm dialect)
       ↓ make_pnm_asm()
   PNM Assembly
       ↓ make_pnm_bin()
   PNM Binary
```

### Key Lowering Transformations (TTIR → TTPNM)

| Triton IR Op | TritonPNM IR Op | Description |
|--------------|-----------------|-------------|
| `tt.load` | `ttpnm.dma_load` | Global → Local memory DMA |
| `tt.store` | `ttpnm.dma_store` | Local → Global memory DMA |
| `tt.dot` | `ttpnm.matmul` | Matrix multiplication on PNM engine |
| `tt.reduce` | `ttpnm.reduce` | Reduction operations |
| `tt.program_id` | `ttpnm.get_cu_id` | Compute unit distribution |

## PNM Dialect Operations

### Memory Operations

- `ttpnm.dma_load` - Load data from global memory to local memory via DMA
- `ttpnm.dma_store` - Store data from local memory to global memory via DMA
- `ttpnm.dma_wait` - Wait for DMA operation completion
- `ttpnm.alloc_local` - Allocate local memory
- `ttpnm.free_local` - Free local memory

### Compute Operations

- `ttpnm.matmul` - Matrix multiplication on PNM matrix engine
- `ttpnm.gemv` - Matrix-vector multiplication
- `ttpnm.elementwise` - Element-wise operations
- `ttpnm.activation` - Activation functions (ReLU, GELU, etc.)
- `ttpnm.reduce` - Reduction operations (sum, max, min)

### Synchronization

- `ttpnm.barrier` - Synchronization barrier
- `ttpnm.fence` - Memory fence

### Control Operations

- `ttpnm.get_cu_id` - Get compute unit ID
- `ttpnm.get_num_cus` - Get total number of compute units

## Usage

### As External Plugin

```bash
export TRITON_PLUGIN_DIRS="/path/to/triton/third_party/pnm"
export TRITON_PNM_ENABLE=1
pip install triton
```

### Writing Kernels

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Standard Triton kernel code
    # Will be compiled for PNM when running on PNM device
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ... rest of matmul implementation
```

## Building

### Build with Triton

```bash
cd triton
pip install -e ".[build]"
```

### Build Dialect Only

```bash
cd triton
mkdir build && cd build
cmake .. -DTRITON_CODEGEN_BACKENDS="pnm"
make TritonPNMIR
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRITON_PNM_ENABLE` | Enable PNM backend | `0` |
| `TRITON_PNM_DEBUG` | Enable debug output | `0` |
| `PNM_ASSEMBLER` | Path to PNM assembler | `pnm-as` |

### Compilation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_compute_units` | int | 4 | Number of PNM compute units |
| `pipeline_depth` | int | 2 | DMA pipeline depth |
| `enable_prefetch` | bool | True | Enable data prefetching |
| `local_mem_size` | int | 256 | Local memory size (KB) |
| `debug` | bool | False | Enable debug mode |

## Testing

```bash
# Run unit tests
pytest third_party/pnm/test/ -v

# Run integration tests (requires PNM hardware/simulator)
TRITON_PNM_INTEGRATION_TESTS=1 pytest third_party/pnm/test/ -v
```

## Development Status

- [x] Backend infrastructure (compiler.py, driver.py)
- [x] Dialect TableGen definitions
- [ ] C++ dialect implementation
- [ ] TritonGPU to PNM lowering passes
- [ ] PNM assembly code generation
- [ ] Hardware integration
- [ ] Performance optimization

## License

MIT License - See LICENSE file for details.
