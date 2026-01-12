# Triton Custom Backend Development Guide

이 문서는 Triton에 NPU/XPU와 같은 Custom Device Backend를 추가하기 위한 분석 결과와 구현 가이드를 제공합니다.

## 목차

1. [Triton 컴파일 파이프라인 개요](#1-triton-컴파일-파이프라인-개요)
2. [프로젝트 구조 분석](#2-프로젝트-구조-분석)
3. [백엔드 시스템 아키텍처](#3-백엔드-시스템-아키텍처)
4. [Custom Backend 구현 요구사항](#4-custom-backend-구현-요구사항)
5. [MLIR 다이얼렉트 구조](#5-mlir-다이얼렉트-구조)
6. [기존 백엔드 분석 (NVIDIA/AMD)](#6-기존-백엔드-분석-nvidiaamd)
7. [구현 로드맵](#7-구현-로드맵)

---

## 1. Triton 컴파일 파이프라인 개요

### 전체 흐름

```
PyTorch torch.compile()
    ↓
TorchDynamo (Python 바이트코드 분석)
    ↓
TorchInductor (그래프 최적화 및 코드 생성)
    ↓
┌──────────────────────────────────────────────────────────────┐
│  Triton Compilation Pipeline                                 │
├──────────────────────────────────────────────────────────────┤
│  Python @triton.jit                                          │
│      ↓                                                       │
│  AST → Triton IR (TTIR)                                      │
│      ↓                                                       │
│  Triton IR → TritonGPU IR (TTGIR)                           │
│      ↓                                                       │
│  TritonGPU IR → LLVM IR                                      │
│      ↓                                                       │
│  LLVM IR → Device Assembly (PTX/AMDGCN/Custom)              │
│      ↓                                                       │
│  Assembly → Binary (cubin/hsaco/custom)                      │
└──────────────────────────────────────────────────────────────┘
    ↓
Device Execution
```

### 컴파일 단계별 설명

| 단계 | IR 형식 | 설명 |
|------|---------|------|
| `ttir` | Triton IR | 하드웨어 독립적인 고수준 IR |
| `ttgir` | TritonGPU IR | GPU 추상화 (워프, 블록, 메모리 레이아웃) |
| `llir` | LLVM IR | LLVM 중간 표현 |
| `asm` | PTX/AMDGCN | 디바이스 어셈블리 |
| `bin` | cubin/hsaco | 실행 가능한 바이너리 |

---

## 2. 프로젝트 구조 분석

### 핵심 디렉토리

```
triton/
├── python/triton/                 # Python 프론트엔드
│   ├── backends/                  # 백엔드 등록 시스템
│   │   ├── __init__.py           # entry_points 기반 백엔드 탐색
│   │   ├── compiler.py           # BaseBackend 추상 클래스 (L23-91)
│   │   └── driver.py             # DriverBase 추상 클래스 (L11-67)
│   ├── compiler/
│   │   ├── compiler.py           # 메인 컴파일 드라이버 (L222-367)
│   │   └── code_generator.py     # AST → TTIR 변환 (73KB)
│   ├── runtime/
│   │   ├── jit.py                # @triton.jit 데코레이터
│   │   ├── autotuner.py          # 자동 튜닝
│   │   └── cache.py              # 컴파일 캐시
│   └── language/                  # Triton DSL
│
├── lib/                           # C++/MLIR 핵심 구현
│   ├── Dialect/
│   │   ├── Triton/               # Triton 다이얼렉트
│   │   │   ├── IR/               # Op/Type 정의
│   │   │   └── Transforms/       # 최적화 패스
│   │   ├── TritonGPU/            # TritonGPU 다이얼렉트
│   │   │   ├── IR/               # GPU 추상화 Op/Type
│   │   │   └── Transforms/       # GPU 최적화 패스
│   │   └── TritonNvidiaGPU/      # NVIDIA 전용 다이얼렉트
│   ├── Conversion/                # 다이얼렉트 변환
│   │   ├── TritonToTritonGPU/    # TTIR → TTGIR
│   │   └── TritonGPUToLLVM/      # TTGIR → LLVM
│   └── Target/LLVMIR/            # LLVM 타겟 코드 생성
│
├── include/triton/                # C++ 헤더
│   └── Dialect/                   # TableGen 정의 (.td 파일)
│
├── third_party/                   # 백엔드 구현
│   ├── nvidia/                    # NVIDIA CUDA 백엔드
│   │   ├── backend/
│   │   │   ├── compiler.py       # CUDABackend (L150-526)
│   │   │   ├── driver.py         # CudaDriver (L716-764)
│   │   │   └── driver.c          # C 런타임
│   │   ├── include/Dialect/      # NVGPU, NVWS 다이얼렉트
│   │   └── lib/                   # C++ 패스 구현
│   └── amd/                       # AMD ROCm 백엔드
│       ├── backend/
│       │   ├── compiler.py       # HIPBackend
│       │   └── driver.py         # HIPDriver
│       └── include/Dialect/      # TritonAMDGPU 다이얼렉트
│
└── setup.py                       # 빌드 및 백엔드 등록 (L606, L748)
```

---

## 3. 백엔드 시스템 아키텍처

### 백엔드 탐색 메커니즘

백엔드는 Python의 `entry_points` 시스템을 통해 자동 탐색됩니다.

```python
# python/triton/backends/__init__.py

def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    for ep in entry_points().select(group="triton.backends"):
        compiler = importlib.import_module(f"{ep.value}.compiler")
        driver = importlib.import_module(f"{ep.value}.driver")
        backends[ep.name] = Backend(
            _find_concrete_subclasses(compiler, BaseBackend),
            _find_concrete_subclasses(driver, DriverBase)
        )
    return backends
```

### 백엔드 등록 (setup.py)

```python
# setup.py:748
entry_points["triton.backends"] = [
    f"{b.name} = triton.backends.{b.name}" for b in backends
]
```

### 컴파일 시 백엔드 선택

```python
# python/triton/compiler/compiler.py:370-375
def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values()
               if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(...)
    return actives[0](target)
```

---

## 4. Custom Backend 구현 요구사항

### 4.1 디렉토리 구조

```
your_xpu_backend/
├── backend/
│   ├── name.conf           # 백엔드 이름 (예: "xpu")
│   ├── __init__.py
│   ├── compiler.py         # [필수] XPUBackend 클래스
│   ├── driver.py           # [필수] XPUDriver 클래스
│   └── driver.c            # [선택] C 런타임 코드
├── include/
│   └── Dialect/            # [선택] 커스텀 MLIR 다이얼렉트
│       └── XPU/
│           └── IR/
│               ├── XPUDialect.td
│               ├── XPUOps.td
│               └── XPUAttrDefs.td
├── lib/                    # [선택] C++ 패스 구현
│   └── Dialect/
│       └── XPU/
│           ├── IR/
│           └── Transforms/
├── language/               # [선택] 디바이스 전용 언어 확장
│   └── xpu/
│       └── __init__.py
└── tools/                  # [선택] 디버깅/프로파일링 도구
```

### 4.2 BaseBackend 구현 (compiler.py)

```python
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from dataclasses import dataclass
from typing import Dict
from types import ModuleType

@dataclass(frozen=True)
class XPUOptions:
    """컴파일 옵션 정의"""
    num_warps: int = 4
    num_stages: int = 2
    # ... 디바이스 특화 옵션들

class XPUBackend(BaseBackend):
    """XPU 백엔드 구현"""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        """이 백엔드가 해당 타겟을 지원하는지 확인

        Args:
            target: GPUTarget(backend='xpu', arch=..., warp_size=...)
        """
        return target.backend == 'xpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "xpubin"  # 최종 바이너리 확장자

    def hash(self) -> str:
        """캐시 키 생성용 고유 식별자"""
        return f'xpu-{self.target.arch}'

    def parse_options(self, opts: dict) -> XPUOptions:
        """컴파일 옵션 파싱 및 검증"""
        return XPUOptions(**{
            k: opts[k] for k in XPUOptions.__dataclass_fields__.keys()
            if k in opts
        })

    def add_stages(self, stages: dict, options: XPUOptions, language: Language) -> None:
        """컴파일 스테이지 정의 - 핵심 메서드!

        각 스테이지는 (src, metadata) -> transformed_src 형태의 함수
        마지막 스테이지는 bytes를 반환해야 함
        """
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)

        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["xpuasm"] = lambda src, metadata: self.make_xpu_asm(src, metadata, options)
        stages["xpubin"] = lambda src, metadata: self.make_xpu_bin(src, metadata, options)

    def load_dialects(self, ctx) -> None:
        """MLIR 컨텍스트에 커스텀 다이얼렉트 로드"""
        # from triton._C.libtriton import xpu
        # xpu.load_dialects(ctx)
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        """디바이스 전용 모듈 매핑 (예: libdevice)"""
        return {}

    def get_codegen_implementation(self, options):
        """코드 생성 헬퍼 함수들"""
        return {
            "convert_custom_types": lambda x: x,
            "min_dot_size": lambda lhs, rhs: (1, 1, 16),
        }

    def pack_metadata(self, metadata):
        """런타임에 전달할 메타데이터 패킹"""
        return (
            metadata.num_warps,
            metadata.shared,
            # ... 기타 필요 정보
        )

    # === 컴파일 스테이지 구현 ===

    def make_ttir(self, mod, metadata, options):
        """Triton IR 최적화"""
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    def make_ttgir(self, mod, metadata, options):
        """TritonGPU IR 변환 및 최적화"""
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            f"xpu:{self.target.arch}",
            options.num_warps,
            self.target.warp_size,
            1  # num_ctas
        )
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        # ... 추가 최적화
        pm.run(mod)
        return mod

    def make_llir(self, mod, metadata, options):
        """LLVM IR 생성"""
        from triton._C.libtriton import ir, passes, llvm

        pm = ir.pass_manager(mod.context)
        # TritonGPU → LLVM 변환 패스
        # xpu.passes.add_to_llvmir(pm, ...)
        pm.run(mod)

        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        # 타겟 설정
        triple = 'xpu64-unknown-unknown'  # 디바이스에 맞게 수정
        llvm.attach_datalayout(llvm_mod, triple, "xpu", "")
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        metadata["shared"] = mod.get_int_attr("ttg.shared")
        return str(llvm_mod)

    def make_xpu_asm(self, src, metadata, options):
        """LLVM IR → XPU 어셈블리"""
        from triton._C.libtriton import llvm

        # LLVM을 사용한 어셈블리 생성
        # 또는 커스텀 컴파일러 호출
        return llvm.translate_to_asm(src, 'xpu64', 'xpu', '', [], True, False)

    def make_xpu_bin(self, src, metadata, options):
        """어셈블리 → 바이너리"""
        import subprocess
        import tempfile

        # 외부 어셈블러 호출 예시
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xpuasm', delete=False) as f:
            f.write(src)
            asm_path = f.name

        bin_path = asm_path + '.bin'
        subprocess.run(['xpu-as', asm_path, '-o', bin_path], check=True)

        with open(bin_path, 'rb') as f:
            return f.read()
```

### 4.3 DriverBase 구현 (driver.py)

```python
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.runtime.build import compile_module_from_src

class XPUDriver(DriverBase):
    """XPU 런타임 드라이버"""

    @classmethod
    def is_active(cls) -> bool:
        """현재 시스템에서 XPU 사용 가능 여부"""
        try:
            import torch
            # 또는 자체 런타임 라이브러리 확인
            return hasattr(torch, 'xpu') and torch.xpu.is_available()
        except ImportError:
            return False

    def __init__(self):
        super().__init__()
        self._init_runtime()

    def _init_runtime(self):
        """런타임 초기화 및 유틸리티 로드"""
        # C 런타임 컴파일 및 로드
        # self.utils = compile_module_from_src(...)
        pass

    def get_current_target(self) -> GPUTarget:
        """현재 디바이스의 GPUTarget 반환"""
        # 실제 디바이스 정보 조회
        arch = self._get_device_arch()
        warp_size = 32  # 디바이스 스펙에 따라
        return GPUTarget("xpu", arch, warp_size)

    def get_active_torch_device(self):
        """활성 PyTorch 디바이스 반환"""
        import torch
        return torch.device("xpu", self.get_current_device())

    def get_current_device(self) -> int:
        """현재 디바이스 인덱스"""
        import torch
        return torch.xpu.current_device()

    def get_benchmarker(self):
        """벤치마크 함수 반환"""
        from triton.testing import do_bench
        return do_bench

    def map_python_to_cpp_type(self, ty: str) -> str:
        """Triton 타입 → C++ 타입 매핑"""
        type_map = {
            "i1": "int8_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "half",
            "bf16": "bfloat16",
            "fp32": "float",
            "fp64": "double",
        }
        if ty[0] == '*':
            return "void*"  # 포인터
        return type_map.get(ty, ty)

    def _get_device_arch(self) -> int:
        """디바이스 아키텍처 버전 조회"""
        # 실제 구현 필요
        return 100


class XPULauncher:
    """커널 런처"""

    def __init__(self, src, metadata):
        self.metadata = metadata
        # 런처 초기화

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        """커널 실행"""
        # 실제 커널 호출 구현
        pass
```

### 4.4 백엔드 등록 방법

#### 방법 A: 외부 플러그인 (권장)

```bash
# 환경 변수로 플러그인 경로 지정
export TRITON_PLUGIN_DIRS="/path/to/your_xpu_backend"

# backend/name.conf 파일 생성
echo "xpu" > /path/to/your_xpu_backend/backend/name.conf

# Triton 설치
pip install triton
```

#### 방법 B: Triton 소스 수정

```python
# setup.py:606 수정
backends = [
    *BackendInstaller.copy(["nvidia", "amd", "xpu"]),  # xpu 추가
    *BackendInstaller.copy_externals()
]
```

---

## 5. MLIR 다이얼렉트 구조

### 5.1 기본 다이얼렉트 계층

```
Triton (tt)           # 하드웨어 독립적
    ↓
TritonGPU (ttg)       # GPU 추상화
    ↓
├── TritonNvidiaGPU   # NVIDIA 특화
├── TritonAMDGPU      # AMD 특화
└── TritonXPU         # 커스텀 (추가 필요)
    ↓
LLVM                  # LLVM IR
```

### 5.2 주요 다이얼렉트 파일 위치

| 다이얼렉트 | 정의 위치 (.td) | 구현 위치 |
|-----------|-----------------|-----------|
| Triton | `include/triton/Dialect/Triton/IR/` | `lib/Dialect/Triton/` |
| TritonGPU | `include/triton/Dialect/TritonGPU/IR/` | `lib/Dialect/TritonGPU/` |
| NVGPU | `third_party/nvidia/include/Dialect/NVGPU/IR/` | `third_party/nvidia/lib/Dialect/` |
| TritonAMDGPU | `third_party/amd/include/Dialect/TritonAMDGPU/IR/` | `third_party/amd/lib/Dialect/` |

### 5.3 TableGen 다이얼렉트 정의 예시

```tablegen
// XPUDialect.td
#ifndef XPU_DIALECT
#define XPU_DIALECT

include "mlir/IR/OpBase.td"

def XPU_Dialect : Dialect {
  let name = "xpu";
  let cppNamespace = "::mlir::triton::xpu";
  let summary = "XPU-specific operations for Triton";
}

#endif
```

```tablegen
// XPUOps.td
#ifndef XPU_OPS
#define XPU_OPS

include "XPUDialect.td"
include "mlir/IR/OpBase.td"

class XPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<XPU_Dialect, mnemonic, traits>;

def XPU_BarrierOp : XPU_Op<"barrier", []> {
  let summary = "XPU barrier synchronization";
  let assemblyFormat = "attr-dict";
}

#endif
```

---

## 6. 기존 백엔드 분석 (NVIDIA/AMD)

### 6.1 NVIDIA 백엔드 컴파일 스테이지

```
ttir:   Triton IR 최적화
        - inliner, canonicalizer, cse
        - rewrite_tensor_pointer

ttgir:  TritonGPU IR 변환 및 최적화
        - convert_to_ttgpuir (워프/블록 매핑)
        - coalesce (메모리 접근 병합)
        - accelerate_matmul (텐서코어 활용)
        - pipeline (소프트웨어 파이프라이닝)
        - prefetch, fence_insertion

llir:   LLVM IR 생성
        - allocate_shared_memory
        - ttgpuir_to_llvmir
        - nvvm_to_llvm

ptx:    PTX 어셈블리 생성
        - LLVM 백엔드 사용

cubin:  바이너리 생성
        - ptxas 외부 도구 호출
```

### 6.2 AMD 백엔드 컴파일 스테이지

```
ttir:   Triton IR 최적화 (NVIDIA와 유사)

ttgir:  TritonGPU IR + AMD 특화
        - AMD 메모리 레이아웃
        - MFMA (Matrix Fused Multiply-Add) 매핑

llir:   LLVM IR 생성
        - AMD GPU 타겟 설정

amdgcn: AMDGCN 어셈블리 생성

hsaco:  바이너리 생성
        - lld 링커 사용
```

### 6.3 주요 차이점

| 항목 | NVIDIA | AMD |
|------|--------|-----|
| 타겟 | cuda:90, cuda:80 등 | hip:gfx940 등 |
| 어셈블리 | PTX | AMDGCN |
| 바이너리 | cubin | hsaco |
| 텐서코어 | WMMA/MMA | MFMA |
| 공유메모리 | shared memory | LDS |
| 동기화 | __syncthreads | __syncthreads |

---

## 7. 구현 로드맵

### Level 1: 최소 구현 (Python-only)

**목표**: 기존 TritonGPU 다이얼렉트 재사용, LLVM IR → 디바이스 코드만 커스텀

1. `backend/compiler.py` 구현
   - `add_stages()`: 기존 ttir/ttgir 패스 재사용
   - `make_llir()`: LLVM IR 생성
   - `make_asm()`, `make_bin()`: 디바이스 특화

2. `backend/driver.py` 구현
   - `is_active()`: 디바이스 감지
   - `get_current_target()`: 타겟 정보

3. 테스트
   ```python
   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, out_ptr, n):
       idx = tl.program_id(0)
       x = tl.load(x_ptr + idx)
       y = tl.load(y_ptr + idx)
       tl.store(out_ptr + idx, x + y)
   ```

### Level 2: 커스텀 다이얼렉트 추가

**목표**: 디바이스 특화 연산 정의

1. TableGen으로 다이얼렉트 정의
   - `include/Dialect/XPU/IR/*.td`

2. C++ 구현
   - `lib/Dialect/XPU/IR/`
   - `lib/Dialect/XPU/Transforms/`

3. CMake 빌드 설정 수정

### Level 3: 전체 최적화 파이프라인

**목표**: 하드웨어 특화 최적화

1. 메모리 레이아웃 최적화
2. 특수 명령어 매핑 (행렬 연산 등)
3. 레지스터 할당 최적화
4. 파이프라이닝

---

## 참고 자료

### 소스 코드 위치

- 백엔드 베이스 클래스: `python/triton/backends/compiler.py:23-91`
- 드라이버 베이스 클래스: `python/triton/backends/driver.py:11-67`
- 컴파일러 드라이버: `python/triton/compiler/compiler.py:222-367`
- NVIDIA 백엔드: `third_party/nvidia/backend/compiler.py`
- AMD 백엔드: `third_party/amd/backend/compiler.py`
- 백엔드 등록: `setup.py:606, 748`

### 관련 문서

- [Triton Documentation](https://triton-lang.org/main/index.html)
- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM Documentation](https://llvm.org/docs/)

---

*이 문서는 Triton 3.5.1 버전을 기준으로 작성되었습니다.*
