# ===----------------------------------------------------------------------=== #
#
# Copyright (c) 2024 Your Organization
# Licensed under the MIT License.
#
# ===----------------------------------------------------------------------=== #
#
# CMake configuration for TritonPNM backend
#
# ===----------------------------------------------------------------------=== #

cmake_minimum_required(VERSION 3.20)

# ===----------------------------------------------------------------------=== #
# Options
# ===----------------------------------------------------------------------=== #

option(TRITON_PNM_BUILD_DIALECT "Build PNM MLIR dialect" ON)
option(TRITON_PNM_BUILD_TESTS "Build PNM tests" OFF)

# ===----------------------------------------------------------------------=== #
# Include directories
# ===----------------------------------------------------------------------=== #

set(TRITON_PNM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${TRITON_PNM_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ===----------------------------------------------------------------------=== #
# TableGen for Dialect definitions
# ===----------------------------------------------------------------------=== #

if(TRITON_PNM_BUILD_DIALECT)
  set(LLVM_TARGET_DEFINITIONS include/Dialect/TritonPNM/IR/TritonPNMOps.td)

  # Generate dialect declarations
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMDialect.h.inc -gen-dialect-decls -dialect=ttpnm)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMDialect.cpp.inc -gen-dialect-defs -dialect=ttpnm)

  # Generate ops declarations
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMOps.h.inc -gen-op-decls)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMOps.cpp.inc -gen-op-defs)

  # Generate attribute declarations
  set(LLVM_TARGET_DEFINITIONS include/Dialect/TritonPNM/IR/TritonPNMAttrDefs.td)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=ttpnm)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ttpnm)

  # Generate type declarations
  set(LLVM_TARGET_DEFINITIONS include/Dialect/TritonPNM/IR/TritonPNMTypes.td)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMTypes.h.inc -gen-typedef-decls -typedefs-dialect=ttpnm)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ttpnm)

  # Generate enum declarations
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMEnums.h.inc -gen-enum-decls)
  mlir_tablegen(include/Dialect/TritonPNM/IR/TritonPNMEnums.cpp.inc -gen-enum-defs)

  add_public_tablegen_target(TritonPNMDialectIncGen)
endif()

# ===----------------------------------------------------------------------=== #
# Library: TritonPNMIR
# ===----------------------------------------------------------------------=== #

if(TRITON_PNM_BUILD_DIALECT)
  add_mlir_dialect_library(TritonPNMIR
    lib/Dialect/TritonPNM/IR/TritonPNMDialect.cpp
    lib/Dialect/TritonPNM/IR/TritonPNMOps.cpp
    lib/Dialect/TritonPNM/IR/TritonPNMTypes.cpp

    DEPENDS
    TritonPNMDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    TritonIR
    TritonGPUIR
  )
endif()

# ===----------------------------------------------------------------------=== #
# Library: TritonPNMTransforms
# ===----------------------------------------------------------------------=== #

if(TRITON_PNM_BUILD_DIALECT)
  add_mlir_library(TritonPNMTransforms
    lib/Dialect/TritonPNM/Transforms/TritonGPUToPNM.cpp

    DEPENDS
    TritonPNMDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    TritonPNMIR
    TritonGPUIR
  )
endif()

# ===----------------------------------------------------------------------=== #
# Library: TritonPNMToLLVM (if using LLVM backend)
# ===----------------------------------------------------------------------=== #

# Uncomment if implementing LLVM-based code generation
# add_mlir_library(TritonPNMToLLVM
#   lib/TritonPNMToLLVM/TritonPNMToLLVMPass.cpp
#
#   DEPENDS
#   TritonPNMDialectIncGen
#
#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRLLVMDialect
#   MLIRLLVMToLLVMIRTranslation
#   TritonPNMIR
# )

# ===----------------------------------------------------------------------=== #
# Python bindings
# ===----------------------------------------------------------------------=== #

if(TRITON_BUILD_PYTHON_MODULE AND TRITON_PNM_BUILD_DIALECT)
  # Generate Python bindings for PNM dialect
  # This will be linked into the main triton module
endif()

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #

if(TRITON_PNM_BUILD_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittest)
endif()
