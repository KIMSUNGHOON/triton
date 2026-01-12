//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TritonPNM dialect for PNM (Processing Near Memory)
// accelerator support in Triton.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PNM_DIALECT
#define TRITON_PNM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TritonPNM Dialect Definition
//===----------------------------------------------------------------------===//

def TritonPNM_Dialect : Dialect {
  let name = "ttpnm";
  let cppNamespace = "::mlir::triton::pnm";

  let summary = "Triton PNM Dialect for Processing-Near-Memory accelerators";

  let description = [{
    The TritonPNM dialect provides operations and types specific to PNM
    (Processing Near Memory) accelerators. This includes:

    - DMA operations for data movement between host and PNM memory
    - PNM-specific compute operations (matrix multiply, element-wise, etc.)
    - Synchronization primitives
    - Memory management operations

    The dialect is designed to be lowered from TritonGPU IR and eventually
    translated to PNM-specific assembly or binary format.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();

    static TritonPNMDialect *getLoaded(MLIRContext *ctx) {
      return ctx->getLoadedDialect<TritonPNMDialect>();
    }
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // TRITON_PNM_DIALECT
