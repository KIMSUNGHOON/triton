# PNM Backend Implementation Approaches Report

**Date**: 2026-01-12
**Author**: Development Team
**Status**: Draft for Review

---

## Executive Summary

This report compares two approaches for implementing the TritonPNM backend:
- **Method A**: C++ MLIR Pass (Standard approach)
- **Method B**: Python Text Parsing (Rapid prototyping approach)

Both methods aim to convert Triton IR (TTIR) to PNM-specific code, but differ significantly in implementation complexity, development time, and long-term maintainability.

---

## Background

### Current Pipeline

```
Triton Kernel (@triton.jit)
        ↓
    Triton IR (TTIR)
        ↓
    PNM Assembly      ← Conversion happens here
        ↓
    PNM Binary
```

The critical question is: **How should we implement the TTIR → PNM Assembly conversion?**

### Why Not Use Existing Infrastructure?

| Infrastructure | Used by | Why PNM Can't Use |
|---------------|---------|-------------------|
| TritonGPU IR | NVIDIA, AMD | GPU concepts (warps, CTAs) don't apply to PNM |
| LLVM Backend | NVIDIA, AMD, Intel | No LLVM backend exists for PNM hardware |
| SPIR-V | Intel XPU | PNM is not a GPU, no SPIR-V support |

---

## Method A: C++ MLIR Pass

### Overview

Implement proper MLIR conversion passes in C++ that transform Triton IR operations to TritonPNM IR operations using MLIR's pattern rewriting infrastructure.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLIR Infrastructure                       │
├─────────────────────────────────────────────────────────────┤
│  Triton IR (TTIR)                                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ tt.load %ptr : !tt.ptr<f32> -> tensor<128xf32>          │ │
│  │ tt.dot %a, %b : tensor<M,K>, tensor<K,N> -> tensor<M,N> │ │
│  │ tt.store %ptr, %val : !tt.ptr<f32>, tensor<128xf32>     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│              C++ ConversionPattern                           │
│              (Type-safe, verified)                           │
│                          ↓                                   │
│  TritonPNM IR (TTPNM)                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ ttpnm.dma_load %ptr : !tt.ptr<f32> -> tensor<128xf32>   │ │
│  │ ttpnm.matmul %a, %b : tensor<M,K>, tensor<K,N> -> ...   │ │
│  │ ttpnm.dma_store %ptr, %val : ...                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Example

```cpp
// lib/Conversion/TritonToPNM/LoadStoreConversion.cpp

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Dialect/TritonPNM/IR/TritonPNMOps.h"

namespace mlir::triton::pnm {

/// Convert tt.load to ttpnm.dma_load
class LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::LoadOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Get operands with type conversion
    Value ptr = adaptor.getPtr();
    Value mask = adaptor.getMask();

    // Verify types
    auto ptrType = ptr.getType().dyn_cast<triton::PointerType>();
    if (!ptrType)
      return failure();

    // Create DMA load operation
    auto dmaOp = rewriter.create<pnm::DMALoadOp>(
        op.getLoc(),
        op.getType(),           // result type
        ptr,                    // source pointer
        /*size=*/nullptr,       // infer from type
        pnm::DMAMode::Sync      // synchronous mode
    );

    // Handle mask if present
    if (mask) {
      // Insert predicated load logic
    }

    rewriter.replaceOp(op, dmaOp.getResult());
    return success();
  }
};

/// Convert tt.store to ttpnm.dma_store
class StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::StoreOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.create<pnm::DMAStoreOp>(
        op.getLoc(),
        adaptor.getValue(),
        adaptor.getPtr(),
        pnm::DMAMode::Sync
    );

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert tt.dot to ttpnm.matmul
class DotOpConversion : public OpConversionPattern<triton::DotOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::DotOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto aType = adaptor.getA().getType().cast<RankedTensorType>();
    auto bType = adaptor.getB().getType().cast<RankedTensorType>();

    // Verify matrix dimensions
    if (aType.getRank() != 2 || bType.getRank() != 2)
      return op.emitError("dot operands must be 2D tensors");

    // Check dimension compatibility: A[M,K] x B[K,N] = C[M,N]
    int64_t K_a = aType.getShape()[1];
    int64_t K_b = bType.getShape()[0];
    if (K_a != K_b)
      return op.emitError("inner dimensions must match");

    auto matmulOp = rewriter.create<pnm::MatMulOp>(
        op.getLoc(),
        op.getType(),
        adaptor.getA(),
        adaptor.getB(),
        adaptor.getC(),  // accumulator
        /*activation=*/nullptr,
        /*transA=*/false,
        /*transB=*/false
    );

    rewriter.replaceOp(op, matmulOp.getResult());
    return success();
  }
};

/// Populate all conversion patterns
void populateTritonToPNMConversionPatterns(
    RewritePatternSet &patterns,
    TypeConverter &typeConverter) {

  patterns.add<
      LoadOpConversion,
      StoreOpConversion,
      DotOpConversion
      // Add more patterns...
  >(typeConverter, patterns.getContext());
}

} // namespace mlir::triton::pnm
```

### Required Files

```
third_party/pnm/
├── include/
│   ├── Dialect/TritonPNM/IR/
│   │   ├── TritonPNMDialect.h      # Dialect class declaration
│   │   ├── TritonPNMOps.h          # Op class declarations
│   │   ├── TritonPNMTypes.h        # Type class declarations
│   │   └── TritonPNMAttrDefs.h     # Attribute declarations
│   │
│   ├── Dialect/TritonPNM/Transforms/
│   │   └── Passes.h                # Optimization passes
│   │
│   └── Conversion/TritonToPNM/
│       └── Passes.h                # Conversion pass declarations
│
├── lib/
│   ├── Dialect/TritonPNM/IR/
│   │   ├── TritonPNMDialect.cpp    # Dialect implementation
│   │   ├── TritonPNMOps.cpp        # Op verifiers, inferType
│   │   └── TritonPNMTypes.cpp      # Type implementation
│   │
│   ├── Dialect/TritonPNM/Transforms/
│   │   ├── OptimizeDMA.cpp         # DMA coalescing, pipelining
│   │   ├── OptimizeMatMul.cpp      # Tiling, scheduling
│   │   └── LocalMemAlloc.cpp       # Memory allocation
│   │
│   └── Conversion/TritonToPNM/
│       ├── TritonToPNMPass.cpp     # Main pass driver
│       ├── LoadStoreConversion.cpp # tt.load/store patterns
│       ├── DotConversion.cpp       # tt.dot patterns
│       ├── ReduceConversion.cpp    # tt.reduce patterns
│       └── ControlConversion.cpp   # program_id patterns
│
└── CMakeLists.txt                  # Build configuration
```

### Estimated Development Time

| Task | Time Estimate |
|------|---------------|
| Header files setup | 1-2 days |
| Dialect implementation | 3-5 days |
| Basic conversion patterns | 5-7 days |
| Testing & debugging | 3-5 days |
| Optimization passes | 5-10 days |
| **Total** | **3-5 weeks** |

### Advantages

1. **Type Safety**: MLIR verifies types at compile time
2. **Maintainability**: Standard MLIR patterns, easy to understand
3. **Extensibility**: Easy to add new ops and patterns
4. **Performance**: Native C++ execution, no string parsing overhead
5. **Debugging**: MLIR provides rich debugging tools
6. **Community**: Standard approach used by all Triton backends

### Disadvantages

1. **Long Setup Time**: Requires full build environment
2. **Compilation Time**: C++ compilation is slow
3. **Learning Curve**: Must understand MLIR APIs
4. **Iteration Speed**: Every change requires recompilation

---

## Method B: Python Text Parsing

### Overview

Parse the MLIR text representation using Python string operations and generate PNM assembly through pattern matching.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Runtime                            │
├─────────────────────────────────────────────────────────────┤
│  Triton IR (TTIR) as Text String                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ "module {                                               │ │
│  │   tt.func @kernel(%arg0: !tt.ptr<f32>) {                │ │
│  │     %0 = tt.load %arg0 : !tt.ptr<f32> -> tensor<128xf32>│ │
│  │     tt.store %arg0, %0 : !tt.ptr<f32>, tensor<128xf32>  │ │
│  │   }                                                     │ │
│  │ }"                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│              Python String Parsing                           │
│              (regex, string match)                           │
│                          ↓                                   │
│  PNM Assembly Text                                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ ".kernel kernel                                         │ │
│  │  dma.load r0, [arg0], 128                               │ │
│  │  dma.store [arg0], r0, 128"                             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Example

```python
# backend/compiler.py

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class TTIROperation:
    """Parsed TTIR operation."""
    op_type: str        # e.g., "tt.load", "tt.dot"
    result: str         # e.g., "%0"
    operands: List[str] # e.g., ["%arg0", "%arg1"]
    attributes: Dict    # e.g., {"axis": 1}
    result_type: str    # e.g., "tensor<128xf32>"

class TTIRParser:
    """Parse TTIR text into structured operations."""

    # Regex patterns for TTIR operations
    PATTERNS = {
        'load': re.compile(
            r'(%\w+)\s*=\s*tt\.load\s+(%\w+).*?->\s*(tensor<[^>]+>)'
        ),
        'store': re.compile(
            r'tt\.store\s+(%\w+),\s*(%\w+)'
        ),
        'dot': re.compile(
            r'(%\w+)\s*=\s*tt\.dot\s+(%\w+),\s*(%\w+)(?:,\s*(%\w+))?.*?->\s*(tensor<[^>]+>)'
        ),
        'reduce': re.compile(
            r'(%\w+)\s*=\s*"tt\.reduce".*?axis\s*=\s*(\d+)'
        ),
        'program_id': re.compile(
            r'(%\w+)\s*=\s*tt\.get_program_id\s+(\w+)'
        ),
        'make_range': re.compile(
            r'(%\w+)\s*=\s*tt\.make_range\s*\{end\s*=\s*(\d+).*?start\s*=\s*(\d+)\}'
        ),
        'splat': re.compile(
            r'(%\w+)\s*=\s*tt\.splat\s+(%\w+)'
        ),
        'addptr': re.compile(
            r'(%\w+)\s*=\s*tt\.addptr\s+(%\w+),\s*(%\w+)'
        ),
        # Arithmetic operations
        'arith_addf': re.compile(r'(%\w+)\s*=\s*arith\.addf\s+(%\w+),\s*(%\w+)'),
        'arith_mulf': re.compile(r'(%\w+)\s*=\s*arith\.mulf\s+(%\w+),\s*(%\w+)'),
        'arith_addi': re.compile(r'(%\w+)\s*=\s*arith\.addi\s+(%\w+),\s*(%\w+)'),
        'arith_muli': re.compile(r'(%\w+)\s*=\s*arith\.muli\s+(%\w+),\s*(%\w+)'),
    }

    def parse(self, mlir_text: str) -> List[TTIROperation]:
        """Parse MLIR text into list of operations."""
        operations = []

        for line in mlir_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith(';'):
                continue

            op = self._parse_line(line)
            if op:
                operations.append(op)

        return operations

    def _parse_line(self, line: str) -> Optional[TTIROperation]:
        """Parse a single MLIR line."""
        for op_type, pattern in self.PATTERNS.items():
            match = pattern.search(line)
            if match:
                return self._create_operation(op_type, match, line)
        return None

    def _create_operation(self, op_type: str, match, line: str) -> TTIROperation:
        """Create TTIROperation from regex match."""
        groups = match.groups()

        if op_type == 'load':
            return TTIROperation(
                op_type='tt.load',
                result=groups[0],
                operands=[groups[1]],
                attributes={},
                result_type=groups[2]
            )
        elif op_type == 'store':
            return TTIROperation(
                op_type='tt.store',
                result='',
                operands=[groups[0], groups[1]],
                attributes={},
                result_type=''
            )
        elif op_type == 'dot':
            operands = [groups[1], groups[2]]
            if groups[3]:  # accumulator
                operands.append(groups[3])
            return TTIROperation(
                op_type='tt.dot',
                result=groups[0],
                operands=operands,
                attributes={},
                result_type=groups[4]
            )
        # ... handle other operations

        return TTIROperation(
            op_type=op_type,
            result=groups[0] if groups else '',
            operands=list(groups[1:]) if len(groups) > 1 else [],
            attributes={},
            result_type=''
        )


class PNMCodeGenerator:
    """Generate PNM assembly from parsed TTIR operations."""

    def __init__(self, options):
        self.options = options
        self.register_counter = 0
        self.value_to_register = {}  # Map MLIR values to PNM registers

    def generate(self, operations: List[TTIROperation]) -> str:
        """Generate PNM assembly from operations."""
        asm_lines = [
            f"; PNM Assembly",
            f"; Generated by Triton PNM Backend (Python)",
            f"",
            f".arch {self.options.arch}",
            f".num_compute_units {self.options.num_compute_units}",
            f".local_mem_size {self.options.local_mem_size}",
            f"",
            f".kernel main",
        ]

        for op in operations:
            asm = self._generate_op(op)
            if asm:
                asm_lines.extend(asm)

        asm_lines.append("")
        asm_lines.append(".end")

        return '\n'.join(asm_lines)

    def _alloc_register(self, value: str) -> str:
        """Allocate a register for an MLIR value."""
        if value not in self.value_to_register:
            reg = f"r{self.register_counter}"
            self.register_counter += 1
            self.value_to_register[value] = reg
        return self.value_to_register[value]

    def _get_register(self, value: str) -> str:
        """Get register for an MLIR value."""
        return self.value_to_register.get(value, value)

    def _generate_op(self, op: TTIROperation) -> List[str]:
        """Generate PNM assembly for a single operation."""

        if op.op_type == 'tt.load':
            dst_reg = self._alloc_register(op.result)
            src_reg = self._get_register(op.operands[0])
            size = self._extract_tensor_size(op.result_type)
            return [
                f"    ; {op.op_type} {op.result} = load {op.operands[0]}",
                f"    dma.load {dst_reg}, [{src_reg}], {size}",
                f"    dma.wait {dst_reg}",
            ]

        elif op.op_type == 'tt.store':
            ptr_reg = self._get_register(op.operands[0])
            val_reg = self._get_register(op.operands[1])
            return [
                f"    ; {op.op_type} store {op.operands[1]} to {op.operands[0]}",
                f"    dma.store [{ptr_reg}], {val_reg}",
                f"    dma.wait",
            ]

        elif op.op_type == 'tt.dot':
            dst_reg = self._alloc_register(op.result)
            a_reg = self._get_register(op.operands[0])
            b_reg = self._get_register(op.operands[1])

            lines = [
                f"    ; {op.op_type} {op.result} = dot({op.operands[0]}, {op.operands[1]})",
            ]

            if len(op.operands) > 2:  # Has accumulator
                acc_reg = self._get_register(op.operands[2])
                lines.append(f"    matmul {dst_reg}, {a_reg}, {b_reg}, {acc_reg}")
            else:
                lines.append(f"    matmul {dst_reg}, {a_reg}, {b_reg}")

            return lines

        elif op.op_type == 'tt.reduce':
            dst_reg = self._alloc_register(op.result)
            src_reg = self._get_register(op.operands[0])
            axis = op.attributes.get('axis', 0)
            return [
                f"    ; {op.op_type} reduce axis={axis}",
                f"    reduce.sum {dst_reg}, {src_reg}, {axis}",
            ]

        elif op.op_type == 'tt.get_program_id':
            dst_reg = self._alloc_register(op.result)
            return [
                f"    ; {op.op_type} get compute unit id",
                f"    get.cu_id {dst_reg}",
            ]

        elif op.op_type.startswith('arith.add'):
            dst_reg = self._alloc_register(op.result)
            a_reg = self._get_register(op.operands[0])
            b_reg = self._get_register(op.operands[1])
            suffix = 'f32' if 'f' in op.op_type else 'i32'
            return [
                f"    add.{suffix} {dst_reg}, {a_reg}, {b_reg}",
            ]

        elif op.op_type.startswith('arith.mul'):
            dst_reg = self._alloc_register(op.result)
            a_reg = self._get_register(op.operands[0])
            b_reg = self._get_register(op.operands[1])
            suffix = 'f32' if 'f' in op.op_type else 'i32'
            return [
                f"    mul.{suffix} {dst_reg}, {a_reg}, {b_reg}",
            ]

        # Unknown operation - emit comment
        return [f"    ; TODO: {op.op_type} not yet implemented"]

    def _extract_tensor_size(self, type_str: str) -> int:
        """Extract total size from tensor type string."""
        # tensor<128x64xf32> -> 128 * 64 = 8192
        match = re.search(r'tensor<([^>]+)>', type_str)
        if match:
            dims_str = match.group(1)
            # Remove element type (e.g., "xf32", "xi32")
            dims_str = re.sub(r'x[a-z]+\d*$', '', dims_str)
            dims = [int(d) for d in dims_str.split('x') if d]
            size = 1
            for d in dims:
                size *= d
            return size
        return 128  # default


class PNMBackend:
    """PNM Backend using Python text parsing."""

    def make_pnm_asm(self, src, metadata: dict, options) -> str:
        """Generate PNM assembly from TTIR using Python parsing."""

        # Convert MLIR module to text
        mlir_text = str(src)

        # Debug output
        if options.debug:
            print("=" * 60)
            print("Input TTIR:")
            print("=" * 60)
            print(mlir_text)
            print("=" * 60)

        # Parse TTIR
        parser = TTIRParser()
        operations = parser.parse(mlir_text)

        if options.debug:
            print(f"Parsed {len(operations)} operations:")
            for op in operations:
                print(f"  {op.op_type}: {op.result} = f({op.operands})")

        # Generate PNM assembly
        codegen = PNMCodeGenerator(options)
        asm_text = codegen.generate(operations)

        if options.debug:
            print("=" * 60)
            print("Generated PNM Assembly:")
            print("=" * 60)
            print(asm_text)
            print("=" * 60)

        return asm_text
```

### Estimated Development Time

| Task | Time Estimate |
|------|---------------|
| Basic parser | 2-4 hours |
| Core operations (load/store/dot) | 4-8 hours |
| Arithmetic operations | 2-4 hours |
| Testing | 2-4 hours |
| **Total** | **1-3 days** |

### Advantages

1. **Rapid Development**: Can implement in hours, not weeks
2. **No Build Required**: Pure Python, instant execution
3. **Fast Iteration**: Change code, run immediately
4. **Easy Debugging**: Print statements, Python debugger
5. **Low Barrier**: No MLIR/C++ expertise needed

### Disadvantages

1. **No Type Safety**: String parsing can't verify types
2. **Fragile**: Regex patterns break on IR format changes
3. **Limited Analysis**: Can't do proper dataflow analysis
4. **Performance**: String parsing is slower than native code
5. **Maintenance**: Hard to extend and maintain long-term
6. **Non-Standard**: Not how other Triton backends work

---

## Comparison Summary

| Aspect | Method A (C++ MLIR) | Method B (Python) |
|--------|---------------------|-------------------|
| **Development Time** | 3-5 weeks | 1-3 days |
| **Type Safety** | Strong | None |
| **Maintainability** | Excellent | Poor |
| **Performance** | Native speed | Python overhead |
| **Debugging** | MLIR tools | Print statements |
| **Extensibility** | Easy | Difficult |
| **Build Requirements** | CMake, LLVM, MLIR | None |
| **Iteration Speed** | Slow (recompile) | Fast (immediate) |
| **Production Ready** | Yes | No |
| **Prototype Ready** | No | Yes |

---

## Recommended Approach

### Phase 1: Python Prototyping (Current)

**Goal**: Validate PNM hardware integration quickly

```
Week 1-2:
├── Implement Python parser for core TTIR ops
├── Generate basic PNM assembly
├── Test on PNM simulator/hardware
└── Identify missing operations and edge cases
```

**Deliverables**:
- Working prototype that compiles simple kernels
- List of all TTIR operations that need support
- Understanding of PNM assembly format requirements

### Phase 2: C++ Implementation (Future)

**Goal**: Production-quality implementation

```
Week 3-8:
├── Port validated Python logic to C++ patterns
├── Implement proper type checking and verification
├── Add optimization passes (DMA coalescing, etc.)
└── Integration testing with full Triton test suite
```

**Deliverables**:
- Complete C++ MLIR conversion pass
- PNM-specific optimization passes
- Full test coverage

### Decision Matrix

| If you need... | Choose |
|----------------|--------|
| Quick hardware validation | Method B (Python) |
| Production deployment | Method A (C++) |
| Understanding IR requirements | Method B first, then A |
| Long-term maintenance | Method A (C++) |
| Single developer | Method B (Python) |
| Team development | Method A (C++) |

---

## Appendix A: TTIR Operations Reference

### Core Operations to Support

| TTIR Operation | Description | Priority |
|----------------|-------------|----------|
| `tt.load` | Load from global memory | P0 |
| `tt.store` | Store to global memory | P0 |
| `tt.dot` | Matrix multiplication | P0 |
| `tt.reduce` | Reduction (sum, max, etc.) | P0 |
| `tt.get_program_id` | Get block/CU index | P0 |
| `tt.make_range` | Create index range | P1 |
| `tt.splat` | Broadcast scalar | P1 |
| `tt.addptr` | Pointer arithmetic | P1 |
| `tt.broadcast` | Broadcast tensor | P1 |
| `tt.expand_dims` | Add dimension | P2 |
| `tt.trans` | Transpose | P2 |

### Arithmetic Operations

| Operation | Description | Priority |
|-----------|-------------|----------|
| `arith.addf/addi` | Addition | P0 |
| `arith.subf/subi` | Subtraction | P0 |
| `arith.mulf/muli` | Multiplication | P0 |
| `arith.divf/divsi/divui` | Division | P1 |
| `arith.cmpf/cmpi` | Comparison | P1 |
| `arith.select` | Conditional select | P1 |
| `arith.maxf/minf` | Max/Min | P1 |

---

## Appendix B: PNM Assembly Format (Draft)

```asm
; PNM Assembly Format Specification (Draft)
; Version: 0.1

; === Directives ===
.arch pnm_v1              ; Target architecture
.num_compute_units 4      ; Number of CUs to use
.local_mem_size 256       ; Local memory in KB

; === Kernel Definition ===
.kernel kernel_name
    ; instructions...
.end

; === Memory Operations ===
dma.load  dst, [src], size    ; Global -> Local
dma.store [dst], src, size    ; Local -> Global
dma.wait  [token]             ; Wait for DMA completion

; === Compute Operations ===
matmul    dst, a, b [, acc]   ; Matrix multiplication
gemv      dst, mat, vec       ; Matrix-vector multiply
reduce.op dst, src, axis      ; Reduction (op = sum|max|min|prod)

; === Arithmetic ===
add.f32   dst, a, b           ; Float add
mul.f32   dst, a, b           ; Float multiply
add.i32   dst, a, b           ; Integer add
mul.i32   dst, a, b           ; Integer multiply

; === Control ===
get.cu_id    dst              ; Get compute unit ID
get.num_cus  dst              ; Get total CU count
barrier                       ; Synchronize all CUs

; === Example Kernel ===
.kernel vector_add
    get.cu_id r0
    mul.i32 r1, r0, 128       ; offset = cu_id * BLOCK_SIZE

    dma.load r2, [arg0 + r1], 128
    dma.load r3, [arg1 + r1], 128
    dma.wait

    add.f32 r4, r2, r3

    dma.store [arg2 + r1], r4, 128
    dma.wait
.end
```

---

## Appendix C: File Changes Summary

### Files Modified

1. `third_party/pnm/backend/compiler.py`
   - Removed `pnmir` stage from pipeline
   - Updated `make_pnm_asm` to work directly from TTIR

2. `third_party/pnm/README.md`
   - Updated pipeline diagram

### Files Created

1. `third_party/pnm/docs/CPP_IMPLEMENTATION_PLAN.md`
   - Detailed C++ implementation plan

2. `third_party/pnm/docs/IMPLEMENTATION_APPROACHES_REPORT.md`
   - This report

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-12 | 0.1 | Dev Team | Initial draft |

---

**End of Report**
