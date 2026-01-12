//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TritonPNM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// Include generated declarations
#include "Dialect/TritonPNM/IR/TritonPNMDialect.h"
#include "Dialect/TritonPNM/IR/TritonPNMDialect.cpp.inc"
#include "Dialect/TritonPNM/IR/TritonPNMEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace pnm {

//===----------------------------------------------------------------------===//
// TritonPNM Dialect
//===----------------------------------------------------------------------===//

void TritonPNMDialect::initialize() {
  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonPNM/IR/TritonPNMOps.cpp.inc"
      >();

  // Register types
  registerTypes();

  // Register attributes
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

void TritonPNMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/TritonPNM/IR/TritonPNMTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void TritonPNMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonPNM/IR/TritonPNMAttrDefs.cpp.inc"
      >();
}

} // namespace pnm
} // namespace triton
} // namespace mlir
