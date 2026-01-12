//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 Your Organization
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TritonPNM operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "Dialect/TritonPNM/IR/TritonPNMDialect.h"
#include "Dialect/TritonPNM/IR/TritonPNMOps.h"

namespace mlir {
namespace triton {
namespace pnm {

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  // Get input types
  auto aType = getA().getType().cast<RankedTensorType>();
  auto bType = getB().getType().cast<RankedTensorType>();
  auto resultType = getResult().getType().cast<RankedTensorType>();

  // Check that inputs are 2D tensors
  if (aType.getRank() != 2 || bType.getRank() != 2) {
    return emitOpError("matmul operands must be 2D tensors");
  }

  // Check dimension compatibility
  // A: [M, K], B: [K, N] -> C: [M, N]
  int64_t aK = aType.getShape()[1];
  int64_t bK = bType.getShape()[0];

  if (aK != bK && aK != ShapedType::kDynamic && bK != ShapedType::kDynamic) {
    return emitOpError("matmul inner dimensions must match");
  }

  // Check result dimensions
  int64_t expectedM = aType.getShape()[0];
  int64_t expectedN = bType.getShape()[1];
  int64_t resultM = resultType.getShape()[0];
  int64_t resultN = resultType.getShape()[1];

  if ((expectedM != resultM && expectedM != ShapedType::kDynamic) ||
      (expectedN != resultN && expectedN != ShapedType::kDynamic)) {
    return emitOpError("result dimensions do not match operand dimensions");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InferTypeOpInterface implementations
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // Get operand types
  auto aType = operands[0].getType().cast<RankedTensorType>();
  auto bType = operands[1].getType().cast<RankedTensorType>();

  // Infer result shape: [M, N]
  int64_t M = aType.getShape()[0];
  int64_t N = bType.getShape()[1];

  // Result element type (might be different for accumulation)
  Type elementType = aType.getElementType();

  // For FP16 inputs, default to FP32 output for precision
  if (elementType.isF16() || elementType.isBF16()) {
    elementType = Float32Type::get(context);
  }

  auto resultType = RankedTensorType::get({M, N}, elementType);
  inferredReturnTypes.push_back(resultType);

  return success();
}

} // namespace pnm
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/TritonPNM/IR/TritonPNMOps.cpp.inc"
