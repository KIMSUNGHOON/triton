//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file defines operations for the TritonPNM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PNM_OPS
#define TRITON_PNM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TritonPNMDialect.td"
include "TritonPNMAttrDefs.td"
include "TritonPNMTypes.td"

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class TritonPNM_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonPNM_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def TritonPNM_AnyTensor : AnyTypeOf<[AnyRankedTensor]>;
def TritonPNM_AnyScalar : AnyTypeOf<[AnyInteger, AnyFloat]>;

//===----------------------------------------------------------------------===//
// Memory Operations - DMA
//===----------------------------------------------------------------------===//

def TritonPNM_DMALoadOp : TritonPNM_Op<"dma_load", [
    MemoryEffects<[MemRead]>
  ]> {
  let summary = "Load data from global memory to PNM local memory via DMA";
  let description = [{
    Initiates a DMA transfer from global memory to PNM local memory.
    This operation can be synchronous or asynchronous depending on the mode.

    Example:
    ```mlir
    // Synchronous load
    %local = ttpnm.dma_load %global_ptr, %size
        {mode = #ttpnm<dma_mode sync>}
        : !tt.ptr<f32>, i64 -> tensor<128x128xf32>

    // Asynchronous load
    %local, %token = ttpnm.dma_load %global_ptr, %size
        {mode = #ttpnm<dma_mode async>}
        : !tt.ptr<f32>, i64 -> tensor<128x128xf32>, !ttpnm.dma_token
    ```
  }];

  let arguments = (ins
    AnyType:$src,
    Optional<I64>:$size,
    OptionalAttr<TritonPNM_DMAMode>:$mode,
    OptionalAttr<I64Attr>:$alignment
  );

  let results = (outs
    TritonPNM_AnyTensor:$result,
    Optional<TritonPNM_DMATokenType>:$token
  );

  let assemblyFormat = [{
    $src (`,` $size^)? attr-dict `:` type($src) (`,` type($size)^)? `->` type($result) (`,` type($token)^)?
  }];
}

def TritonPNM_DMAStoreOp : TritonPNM_Op<"dma_store", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "Store data from PNM local memory to global memory via DMA";
  let description = [{
    Initiates a DMA transfer from PNM local memory to global memory.

    Example:
    ```mlir
    ttpnm.dma_store %local_data, %global_ptr
        {mode = #ttpnm<dma_mode sync>}
        : tensor<128x128xf32>, !tt.ptr<f32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$src,
    AnyType:$dst,
    OptionalAttr<TritonPNM_DMAMode>:$mode
  );

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

def TritonPNM_DMAWaitOp : TritonPNM_Op<"dma_wait", []> {
  let summary = "Wait for DMA operation to complete";
  let description = [{
    Waits for an asynchronous DMA operation to complete.
    Takes a DMA token and blocks until the transfer is finished.

    Example:
    ```mlir
    ttpnm.dma_wait %token : !ttpnm.dma_token
    ```
  }];

  let arguments = (ins TritonPNM_DMATokenType:$token);

  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//===----------------------------------------------------------------------===//
// Memory Operations - Local Memory Management
//===----------------------------------------------------------------------===//

def TritonPNM_AllocLocalOp : TritonPNM_Op<"alloc_local", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let summary = "Allocate PNM local memory";
  let description = [{
    Allocates memory in PNM's local (near-memory) storage.

    Example:
    ```mlir
    %local_buf = ttpnm.alloc_local : tensor<128x128xf32>
    ```
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$alignment
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TritonPNM_FreeLocalOp : TritonPNM_Op<"free_local", [
    MemoryEffects<[MemFree]>
  ]> {
  let summary = "Free PNM local memory";

  let arguments = (ins TritonPNM_AnyTensor:$memref);

  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// Compute Operations - Matrix Operations
//===----------------------------------------------------------------------===//

def TritonPNM_MatMulOp : TritonPNM_Op<"matmul", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PNM matrix multiplication";
  let description = [{
    Performs matrix multiplication on PNM's matrix engine.
    Supports optional accumulator and activation function.

    C = activation(A @ B + acc)

    Example:
    ```mlir
    // Basic matmul
    %c = ttpnm.matmul %a, %b : tensor<128x64xf16>, tensor<64x128xf16> -> tensor<128x128xf32>

    // Matmul with accumulator
    %c = ttpnm.matmul %a, %b, %acc
        : tensor<128x64xf16>, tensor<64x128xf16>, tensor<128x128xf32> -> tensor<128x128xf32>

    // Matmul with activation
    %c = ttpnm.matmul %a, %b {activation = #ttpnm<activation_type relu>}
        : tensor<128x64xf16>, tensor<64x128xf16> -> tensor<128x128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$a,
    TritonPNM_AnyTensor:$b,
    Optional<TritonPNM_AnyTensor>:$acc,
    OptionalAttr<TritonPNM_ActivationType>:$activation,
    OptionalAttr<BoolAttr>:$transA,
    OptionalAttr<BoolAttr>:$transB
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = [{
    $a `,` $b (`,` $acc^)? attr-dict `:`
    type($a) `,` type($b) (`,` type($acc)^)? `->` type($result)
  }];

  let hasVerifier = 1;
}

def TritonPNM_GEMVOp : TritonPNM_Op<"gemv", [Pure]> {
  let summary = "PNM matrix-vector multiplication";
  let description = [{
    Performs matrix-vector multiplication: y = A @ x + b

    Example:
    ```mlir
    %y = ttpnm.gemv %A, %x : tensor<128x64xf16>, tensor<64xf16> -> tensor<128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$matrix,
    TritonPNM_AnyTensor:$vector,
    Optional<TritonPNM_AnyTensor>:$bias
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = [{
    $matrix `,` $vector (`,` $bias^)? attr-dict `:`
    type($matrix) `,` type($vector) (`,` type($bias)^)? `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Compute Operations - Element-wise Operations
//===----------------------------------------------------------------------===//

def TritonPNM_ElementwiseOp : TritonPNM_Op<"elementwise", [
    Pure,
    SameOperandsAndResultType
  ]> {
  let summary = "PNM element-wise operation";
  let description = [{
    Performs element-wise operations on PNM.
    Supports various activation functions and binary operations.

    Example:
    ```mlir
    %out = ttpnm.elementwise %a {op = "relu"} : tensor<128xf32>
    %out = ttpnm.elementwise %a, %b {op = "add"} : tensor<128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$input1,
    Optional<TritonPNM_AnyTensor>:$input2,
    StrAttr:$op
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = [{
    $input1 (`,` $input2^)? attr-dict `:` type($input1)
  }];
}

def TritonPNM_ActivationOp : TritonPNM_Op<"activation", [
    Pure,
    SameOperandsAndResultType
  ]> {
  let summary = "PNM activation function";
  let description = [{
    Applies an activation function on PNM hardware.

    Example:
    ```mlir
    %out = ttpnm.activation %in {type = #ttpnm<activation_type gelu>}
        : tensor<128x128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$input,
    TritonPNM_ActivationType:$type
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// Compute Operations - Reduction Operations
//===----------------------------------------------------------------------===//

def TritonPNM_ReduceOp : TritonPNM_Op<"reduce", [Pure]> {
  let summary = "PNM reduction operation";
  let description = [{
    Performs reduction operations (sum, max, min, etc.) on PNM.

    Example:
    ```mlir
    %sum = ttpnm.reduce %input {op = "sum", axis = 1}
        : tensor<128x64xf32> -> tensor<128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$input,
    StrAttr:$op,
    I64Attr:$axis
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def TritonPNM_BarrierOp : TritonPNM_Op<"barrier", []> {
  let summary = "PNM synchronization barrier";
  let description = [{
    Synchronizes all compute units within a PNM device.

    Example:
    ```mlir
    ttpnm.barrier
    ttpnm.barrier {scope = "device"}
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$scope
  );

  let assemblyFormat = "attr-dict";
}

def TritonPNM_FenceOp : TritonPNM_Op<"fence", []> {
  let summary = "Memory fence operation";
  let description = [{
    Ensures memory ordering for PNM operations.

    Example:
    ```mlir
    ttpnm.fence {memory_space = #ttpnm<memory_space local>}
    ```
  }];

  let arguments = (ins
    OptionalAttr<TritonPNM_MemorySpace>:$memory_space
  );

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Control Operations
//===----------------------------------------------------------------------===//

def TritonPNM_GetComputeUnitIdOp : TritonPNM_Op<"get_cu_id", [Pure]> {
  let summary = "Get current compute unit ID";
  let description = [{
    Returns the ID of the current PNM compute unit.
    Similar to threadIdx/blockIdx in CUDA.

    Example:
    ```mlir
    %cu_id = ttpnm.get_cu_id : i32
    ```
  }];

  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TritonPNM_GetNumComputeUnitsOp : TritonPNM_Op<"get_num_cus", [Pure]> {
  let summary = "Get total number of compute units";

  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def TritonPNM_BroadcastOp : TritonPNM_Op<"broadcast", [Pure]> {
  let summary = "Broadcast data across compute units";
  let description = [{
    Broadcasts data from one compute unit to all others.

    Example:
    ```mlir
    %bcast = ttpnm.broadcast %data {root = 0}
        : tensor<128xf32> -> tensor<128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$input,
    I32Attr:$root
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def TritonPNM_AllReduceOp : TritonPNM_Op<"all_reduce", [Pure]> {
  let summary = "All-reduce across compute units";
  let description = [{
    Performs an all-reduce operation across all compute units.

    Example:
    ```mlir
    %reduced = ttpnm.all_reduce %data {op = "sum"}
        : tensor<128xf32> -> tensor<128xf32>
    ```
  }];

  let arguments = (ins
    TritonPNM_AnyTensor:$input,
    StrAttr:$op
  );

  let results = (outs TritonPNM_AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // TRITON_PNM_OPS
