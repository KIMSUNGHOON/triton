//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file defines attributes for the TritonPNM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PNM_ATTR_DEFS
#define TRITON_PNM_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "TritonPNMDialect.td"

//===----------------------------------------------------------------------===//
// TritonPNM Attribute Definitions
//===----------------------------------------------------------------------===//

class TritonPNM_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TritonPNM_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Memory Space Enum
//===----------------------------------------------------------------------===//

def TritonPNM_MemorySpace : I32EnumAttr<
    "MemorySpace", "PNM memory space",
    [
      I32EnumAttrCase<"HOST", 0, "host">,
      I32EnumAttrCase<"GLOBAL", 1, "global">,
      I32EnumAttrCase<"LOCAL", 2, "local">,
      I32EnumAttrCase<"REGISTER", 3, "register">
    ]> {
  let cppNamespace = "::mlir::triton::pnm";
}

//===----------------------------------------------------------------------===//
// DMA Transfer Mode
//===----------------------------------------------------------------------===//

def TritonPNM_DMAMode : I32EnumAttr<
    "DMAMode", "DMA transfer mode",
    [
      I32EnumAttrCase<"SYNC", 0, "sync">,
      I32EnumAttrCase<"ASYNC", 1, "async">,
      I32EnumAttrCase<"BULK", 2, "bulk">
    ]> {
  let cppNamespace = "::mlir::triton::pnm";
}

//===----------------------------------------------------------------------===//
// Compute Operation Type
//===----------------------------------------------------------------------===//

def TritonPNM_ComputeType : I32EnumAttr<
    "ComputeType", "PNM compute operation type",
    [
      I32EnumAttrCase<"GEMM", 0, "gemm">,
      I32EnumAttrCase<"GEMV", 1, "gemv">,
      I32EnumAttrCase<"ELEMENTWISE", 2, "elementwise">,
      I32EnumAttrCase<"REDUCE", 3, "reduce">,
      I32EnumAttrCase<"ACTIVATION", 4, "activation">
    ]> {
  let cppNamespace = "::mlir::triton::pnm";
}

//===----------------------------------------------------------------------===//
// Activation Function Type
//===----------------------------------------------------------------------===//

def TritonPNM_ActivationType : I32EnumAttr<
    "ActivationType", "Activation function type",
    [
      I32EnumAttrCase<"NONE", 0, "none">,
      I32EnumAttrCase<"RELU", 1, "relu">,
      I32EnumAttrCase<"GELU", 2, "gelu">,
      I32EnumAttrCase<"SILU", 3, "silu">,
      I32EnumAttrCase<"SIGMOID", 4, "sigmoid">,
      I32EnumAttrCase<"TANH", 5, "tanh">
    ]> {
  let cppNamespace = "::mlir::triton::pnm";
}

//===----------------------------------------------------------------------===//
// Data Layout Attribute
//===----------------------------------------------------------------------===//

def TritonPNM_DataLayoutAttr : TritonPNM_Attr<"DataLayout", "data_layout"> {
  let summary = "PNM data layout attribute";
  let description = [{
    Describes the data layout for PNM operations.
    Includes information about tiling, padding, and memory alignment.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tileShape,
    "int64_t":$alignment,
    "bool":$isPadded
  );

  let assemblyFormat = "`<` `tile` `=` `[` $tileShape `]` `,` `align` `=` $alignment (`,` `padded` $isPadded^)? `>`";
}

//===----------------------------------------------------------------------===//
// PNM Execution Config Attribute
//===----------------------------------------------------------------------===//

def TritonPNM_ExecConfigAttr : TritonPNM_Attr<"ExecConfig", "exec_config"> {
  let summary = "PNM execution configuration";
  let description = [{
    Configuration for PNM kernel execution including:
    - Number of compute units to use
    - Memory allocation strategy
    - Pipelining depth
  }];

  let parameters = (ins
    "int64_t":$numComputeUnits,
    "int64_t":$pipelineDepth,
    "bool":$enablePrefetch
  );

  let assemblyFormat = "`<` `cu` `=` $numComputeUnits `,` `pipeline` `=` $pipelineDepth (`,` `prefetch` $enablePrefetch^)? `>`";
}

#endif // TRITON_PNM_ATTR_DEFS
