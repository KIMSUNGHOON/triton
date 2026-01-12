# TritonPNM C++ Dialect Implementation Plan

## Overview

This document outlines the complete C++ implementation required for the TritonPNM backend.

## Architecture: Triton IR → TritonPNM IR (Direct Lowering)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Triton IR (ttir)                                  │
│  Operations: tt.load, tt.store, tt.dot, tt.reduce, tt.program_id, etc.  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼ ConvertTritonToPNM Pass (NEW)
┌─────────────────────────────────────────────────────────────────────────┐
│                       TritonPNM IR (ttpnm)                              │
│  Operations: ttpnm.dma_load, ttpnm.dma_store, ttpnm.matmul, etc.       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼ PNMToASM Pass (NEW)
┌─────────────────────────────────────────────────────────────────────────┐
│                        PNM Assembly                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼ Assembler
┌─────────────────────────────────────────────────────────────────────────┐
│                        PNM Binary                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
third_party/pnm/
├── include/
│   ├── Dialect/TritonPNM/IR/
│   │   ├── TritonPNMDialect.h         [STUB - needs implementation]
│   │   ├── TritonPNMOps.h             [STUB - needs implementation]
│   │   ├── TritonPNMTypes.h           [STUB - needs implementation]
│   │   ├── TritonPNMAttrDefs.h        [STUB - needs implementation]
│   │   └── *.td                        [DONE]
│   │
│   ├── Dialect/TritonPNM/Transforms/
│   │   └── Passes.h                   [NEW - needs creation]
│   │
│   ├── Conversion/TritonToPNM/
│   │   └── Passes.h                   [NEW - needs creation]
│   │
│   └── TritonPNMToASM/
│       └── TritonPNMToASMPass.h       [NEW - needs creation]
│
├── lib/
│   ├── Dialect/TritonPNM/IR/
│   │   ├── TritonPNMDialect.cpp       [EXISTS - needs completion]
│   │   ├── TritonPNMOps.cpp           [EXISTS - needs completion]
│   │   └── TritonPNMTypes.cpp         [NEW - needs creation]
│   │
│   ├── Dialect/TritonPNM/Transforms/
│   │   ├── OptimizeDMA.cpp            [NEW - PNM-specific optimizations]
│   │   ├── OptimizeMatMul.cpp         [NEW - MatMul tiling/scheduling]
│   │   └── LocalMemoryAllocation.cpp  [NEW - Local memory management]
│   │
│   ├── Conversion/TritonToPNM/
│   │   ├── TritonToPNMPass.cpp        [NEW - Main lowering pass]
│   │   ├── LoadStoreConversion.cpp    [NEW - tt.load/store -> DMA]
│   │   ├── DotConversion.cpp          [NEW - tt.dot -> matmul]
│   │   ├── ReduceConversion.cpp       [NEW - tt.reduce -> reduce]
│   │   └── ControlConversion.cpp      [NEW - program_id -> cu_id]
│   │
│   └── TritonPNMToASM/
│       ├── TritonPNMToASMPass.cpp     [NEW - Code generation]
│       └── PNMAsmPrinter.cpp          [NEW - Assembly emission]
│
└── CMakeLists.txt                      [EXISTS - needs update]
```

---

## Implementation Priority & Tasks

### Phase 1: Core Dialect Infrastructure (Required)

#### 1.1 Header Files (include/Dialect/TritonPNM/IR/)

| File | Status | Implementation Needed |
|------|--------|----------------------|
| `TritonPNMDialect.h` | STUB | Add `#include` for generated `.inc` files |
| `TritonPNMOps.h` | STUB | Add operation class declarations |
| `TritonPNMTypes.h` | STUB | Add type class declarations |
| `TritonPNMAttrDefs.h` | STUB | Add attribute class declarations |

**TritonPNMDialect.h** - Template:
```cpp
#ifndef TRITON_PNM_IR_DIALECT_H
#define TRITON_PNM_IR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

// Include generated dialect declarations
#include "Dialect/TritonPNM/IR/TritonPNMDialect.h.inc"

#endif // TRITON_PNM_IR_DIALECT_H
```

**TritonPNMOps.h** - Template:
```cpp
#ifndef TRITON_PNM_IR_OPS_H
#define TRITON_PNM_IR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/TritonPNM/IR/TritonPNMDialect.h"
#include "Dialect/TritonPNM/IR/TritonPNMTypes.h"
#include "Dialect/TritonPNM/IR/TritonPNMAttrDefs.h"

// Include generated enum declarations
#include "Dialect/TritonPNM/IR/TritonPNMEnums.h.inc"

// Include generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonPNM/IR/TritonPNMAttrDefs.h.inc"

// Include generated op declarations
#define GET_OP_CLASSES
#include "Dialect/TritonPNM/IR/TritonPNMOps.h.inc"

#endif // TRITON_PNM_IR_OPS_H
```

#### 1.2 Source Files (lib/Dialect/TritonPNM/IR/)

| File | Status | Methods to Implement |
|------|--------|---------------------|
| `TritonPNMDialect.cpp` | EXISTS | `initialize()` - already done |
| `TritonPNMOps.cpp` | EXISTS | `verify()` for each op with `hasVerifier=1` |
| `TritonPNMTypes.cpp` | NEW | Type parsing/printing (auto-generated mostly) |

**Operations requiring verify() implementation:**
1. `MatMulOp::verify()` - Check dimension compatibility (DONE)
2. `GEMVOp::verify()` - Check matrix/vector dimensions
3. `ReduceOp::verify()` - Check axis validity
4. `DMALoadOp::verify()` - Check pointer/tensor compatibility
5. `DMAStoreOp::verify()` - Check type compatibility

---

### Phase 2: Triton to PNM Conversion Pass (Critical)

This is the **most important** part - converting Triton IR to TritonPNM IR.

#### 2.1 Pass Registration Header

**include/Conversion/TritonToPNM/Passes.h**:
```cpp
#ifndef TRITON_CONVERSION_TRITONTOPNM_PASSES_H
#define TRITON_CONVERSION_TRITONTOPNM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createConvertTritonToPNMPass();

// Pass registration
#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToPNM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
```

#### 2.2 Conversion Patterns

**lib/Conversion/TritonToPNM/TritonToPNMPass.cpp**:

| Triton IR Op | TritonPNM IR Op | Conversion Logic |
|--------------|-----------------|------------------|
| `tt.load` | `ttpnm.dma_load` | Global→Local DMA |
| `tt.store` | `ttpnm.dma_store` | Local→Global DMA |
| `tt.dot` | `ttpnm.matmul` | Matrix engine dispatch |
| `tt.reduce` | `ttpnm.reduce` | Reduction unit dispatch |
| `tt.program_id` | `ttpnm.get_cu_id` | CU distribution |
| `tt.get_num_programs` | `ttpnm.get_num_cus` | CU count |
| `tt.broadcast` | `ttpnm.broadcast` | Inter-CU broadcast |
| `tt.make_range` | Keep or lower | Index generation |
| `tt.splat` | Keep or lower | Value broadcast |

**Key Conversion Pattern Example**:
```cpp
class LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::LoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // 1. Allocate local memory for result
    auto allocOp = rewriter.create<pnm::AllocLocalOp>(
        op.getLoc(), op.getType());

    // 2. Create DMA load from global to local
    auto dmaOp = rewriter.create<pnm::DMALoadOp>(
        op.getLoc(),
        op.getType(),
        adaptor.getPtr(),
        /*size=*/nullptr,
        pnm::DMAMode::Sync);

    rewriter.replaceOp(op, dmaOp.getResult());
    return success();
  }
};
```

#### 2.3 Required Conversion Files

| File | Purpose | Key Patterns |
|------|---------|--------------|
| `LoadStoreConversion.cpp` | Memory ops | `LoadOp`, `StoreOp`, `AtomicOp` |
| `DotConversion.cpp` | Matrix ops | `DotOp`, `DotScaledOp` |
| `ReduceConversion.cpp` | Reductions | `ReduceOp`, `ScanOp` |
| `ControlConversion.cpp` | Control flow | `ProgramIdOp`, `GetNumProgramsOp` |
| `ElementwiseConversion.cpp` | Elementwise | `AddOp`, `SubOp`, `MulOp`, etc. |

---

### Phase 3: PNM-Specific Optimization Passes

#### 3.1 DMA Optimization (lib/Dialect/TritonPNM/Transforms/OptimizeDMA.cpp)

**Optimizations:**
1. **DMA Coalescing** - Merge adjacent DMA operations
2. **DMA Pipelining** - Overlap DMA with compute
3. **Prefetch Insertion** - Add prefetch hints
4. **Double Buffering** - Ping-pong buffer pattern

```cpp
// Example: DMA Pipelining Pattern
// Before:
//   dma_load A0
//   dma_wait
//   compute A0
//   dma_load A1
//   dma_wait
//   compute A1
//
// After:
//   dma_load A0 (async)
//   dma_load A1 (async)
//   dma_wait A0
//   compute A0
//   dma_wait A1
//   compute A1
```

#### 3.2 MatMul Optimization (lib/Dialect/TritonPNM/Transforms/OptimizeMatMul.cpp)

**Optimizations:**
1. **Tiling** - Break large matmul into tiles fitting local memory
2. **Scheduling** - Optimize tile execution order
3. **Fusion** - Fuse matmul with bias/activation

#### 3.3 Local Memory Allocation (lib/Dialect/TritonPNM/Transforms/LocalMemoryAllocation.cpp)

**Tasks:**
1. **Liveness Analysis** - Track tensor lifetimes
2. **Memory Planning** - Assign local memory offsets
3. **Spilling** - Handle overflow to global memory

---

### Phase 4: PNM Assembly Code Generation

#### 4.1 PNMToASM Pass (lib/TritonPNMToASM/)

| Component | Purpose |
|-----------|---------|
| `TritonPNMToASMPass.cpp` | Main code generation pass |
| `PNMAsmPrinter.cpp` | Assembly text emission |

**Assembly format (placeholder - depends on actual PNM ISA):**
```asm
; PNM Assembly Example
.kernel matmul_kernel
    ; DMA load A tile
    dma.load r0, [global_a + offset], 128
    dma.wait r0

    ; DMA load B tile
    dma.load r1, [global_b + offset], 128
    dma.wait r1

    ; Matrix multiply
    matmul r2, r0, r1, 128, 64, 128

    ; DMA store result
    dma.store [global_c + offset], r2, 128
    dma.wait

    ; Barrier
    barrier.device
```

---

## CMakeLists.txt Updates

```cmake
# Add new libraries

# TritonPNMIR (update existing)
add_mlir_dialect_library(TritonPNMIR
  lib/Dialect/TritonPNM/IR/TritonPNMDialect.cpp
  lib/Dialect/TritonPNM/IR/TritonPNMOps.cpp
  lib/Dialect/TritonPNM/IR/TritonPNMTypes.cpp    # NEW
  ...
)

# TritonPNMTransforms (NEW)
add_mlir_library(TritonPNMTransforms
  lib/Dialect/TritonPNM/Transforms/OptimizeDMA.cpp
  lib/Dialect/TritonPNM/Transforms/OptimizeMatMul.cpp
  lib/Dialect/TritonPNM/Transforms/LocalMemoryAllocation.cpp
  ...
)

# TritonToPNMConversion (NEW)
add_mlir_library(TritonToPNMConversion
  lib/Conversion/TritonToPNM/TritonToPNMPass.cpp
  lib/Conversion/TritonToPNM/LoadStoreConversion.cpp
  lib/Conversion/TritonToPNM/DotConversion.cpp
  lib/Conversion/TritonToPNM/ReduceConversion.cpp
  lib/Conversion/TritonToPNM/ControlConversion.cpp
  ...
)

# TritonPNMToASM (NEW)
add_mlir_library(TritonPNMToASM
  lib/TritonPNMToASM/TritonPNMToASMPass.cpp
  lib/TritonPNMToASM/PNMAsmPrinter.cpp
  ...
)
```

---

## Implementation Checklist

### Phase 1: Core Dialect (Week 1-2)
- [ ] Create `TritonPNMDialect.h`
- [ ] Create `TritonPNMOps.h`
- [ ] Create `TritonPNMTypes.h`
- [ ] Create `TritonPNMAttrDefs.h`
- [ ] Complete `TritonPNMOps.cpp` - all verify() methods
- [ ] Create `TritonPNMTypes.cpp`
- [ ] Test dialect registration and op creation

### Phase 2: Conversion Pass (Week 3-4)
- [ ] Create `include/Conversion/TritonToPNM/Passes.h`
- [ ] Create `lib/Conversion/TritonToPNM/TritonToPNMPass.cpp`
- [ ] Implement `LoadStoreConversion.cpp`
- [ ] Implement `DotConversion.cpp`
- [ ] Implement `ReduceConversion.cpp`
- [ ] Implement `ControlConversion.cpp`
- [ ] Test simple kernel lowering

### Phase 3: Optimization Passes (Week 5-6)
- [ ] Implement `OptimizeDMA.cpp`
- [ ] Implement `OptimizeMatMul.cpp`
- [ ] Implement `LocalMemoryAllocation.cpp`
- [ ] Test optimization correctness

### Phase 4: Code Generation (Week 7-8)
- [ ] Define PNM assembly format
- [ ] Implement `TritonPNMToASMPass.cpp`
- [ ] Implement `PNMAsmPrinter.cpp`
- [ ] Integration testing

---

## Dependencies

| Dependency | Reason |
|------------|--------|
| MLIR Core | IR infrastructure |
| Triton Dialect | Source IR to lower from |
| LLVM Support | Utilities |

**NOT Required:**
- TritonGPU Dialect (we bypass it)
- LLVM Backend (we generate PNM assembly directly)
- CUDA/HIP SDK

---

## Testing Strategy

1. **Unit Tests** - Test each operation in isolation
2. **Lit Tests** - Test IR transformations
3. **Integration Tests** - Test full compilation pipeline
4. **Hardware Tests** - Test on actual PNM device (when available)

```bash
# Run dialect tests
pytest third_party/pnm/test/test_pnm_backend.py -v

# Run lit tests (when available)
llvm-lit third_party/pnm/test/lit/
```

---

## Next Steps

1. **Immediate**: Create header files (`TritonPNMDialect.h`, etc.)
2. **Short-term**: Implement Triton to PNM conversion pass
3. **Medium-term**: Implement PNM-specific optimizations
4. **Long-term**: Implement assembly code generation

Start with Phase 1 to ensure the dialect infrastructure is solid before moving to conversion passes.
