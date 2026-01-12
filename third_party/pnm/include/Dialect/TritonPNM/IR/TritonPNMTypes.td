//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file defines types for the TritonPNM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PNM_TYPES
#define TRITON_PNM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TritonPNMDialect.td"

//===----------------------------------------------------------------------===//
// TritonPNM Type Definitions
//===----------------------------------------------------------------------===//

class TritonPNM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TritonPNM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PNM Memory Types
//===----------------------------------------------------------------------===//

def TritonPNM_LocalMemRefType : TritonPNM_Type<"LocalMemRef", "local_memref"> {
  let summary = "PNM local memory reference type";
  let description = [{
    Represents a reference to data stored in PNM's local memory (near-memory).
    This is the fast memory close to the PNM compute units.

    Example:
    ```mlir
    %local = ttpnm.alloc_local : !ttpnm.local_memref<128x128xf32>
    ```
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let assemblyFormat = "`<` custom<ShapeAndType>($shape, $elementType) `>`";
}

def TritonPNM_DMATokenType : TritonPNM_Type<"DMAToken", "dma_token"> {
  let summary = "DMA operation token for synchronization";
  let description = [{
    A token type representing an in-flight DMA operation.
    Used to track and synchronize DMA transfers.

    Example:
    ```mlir
    %token = ttpnm.dma_start_load %src : tensor<128xf32> -> !ttpnm.dma_token
    ttpnm.dma_wait %token : !ttpnm.dma_token
    ```
  }];
}

def TritonPNM_ComputeUnitIdType : TritonPNM_Type<"ComputeUnitId", "cu_id"> {
  let summary = "PNM Compute Unit identifier type";
  let description = [{
    Represents the ID of a PNM compute unit.
    Used for work distribution across multiple PNM units.
  }];
}

#endif // TRITON_PNM_TYPES
