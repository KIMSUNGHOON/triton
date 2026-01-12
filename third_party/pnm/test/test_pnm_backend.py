"""
Test suite for the TritonPNM backend.

These tests verify the basic functionality of the PNM backend.
Run with: pytest third_party/pnm/test/test_pnm_backend.py
"""

import pytest
import os

# Enable PNM backend for testing
os.environ['TRITON_PNM_ENABLE'] = '1'


class TestPNMBackendImport:
    """Test that PNM backend modules can be imported."""

    def test_import_compiler(self):
        """Test importing the compiler module."""
        from triton.backends.pnm.compiler import PNMBackend, PNMOptions
        assert PNMBackend is not None
        assert PNMOptions is not None

    def test_import_driver(self):
        """Test importing the driver module."""
        from triton.backends.pnm.driver import PNMDriver, PNMLauncher
        assert PNMDriver is not None
        assert PNMLauncher is not None


class TestPNMOptions:
    """Test PNM compilation options."""

    def test_default_options(self):
        """Test default option values."""
        from triton.backends.pnm.compiler import PNMOptions

        opts = PNMOptions()
        assert opts.num_compute_units == 4
        assert opts.pipeline_depth == 2
        assert opts.enable_prefetch == True
        assert opts.backend_name == 'pnm'

    def test_custom_options(self):
        """Test custom option values."""
        from triton.backends.pnm.compiler import PNMOptions

        opts = PNMOptions(
            num_compute_units=8,
            pipeline_depth=4,
            enable_prefetch=False
        )
        assert opts.num_compute_units == 8
        assert opts.pipeline_depth == 4
        assert opts.enable_prefetch == False

    def test_options_hash(self):
        """Test that options generate consistent hashes."""
        from triton.backends.pnm.compiler import PNMOptions

        opts1 = PNMOptions(num_compute_units=4)
        opts2 = PNMOptions(num_compute_units=4)
        opts3 = PNMOptions(num_compute_units=8)

        assert opts1.hash() == opts2.hash()
        assert opts1.hash() != opts3.hash()


class TestPNMBackend:
    """Test PNM backend functionality."""

    def test_supports_target(self):
        """Test target support check."""
        from triton.backends.pnm.compiler import PNMBackend
        from triton.backends.compiler import GPUTarget

        pnm_target = GPUTarget(backend='pnm', arch='pnm_v1', warp_size=32)
        cuda_target = GPUTarget(backend='cuda', arch=90, warp_size=32)

        assert PNMBackend.supports_target(pnm_target) == True
        assert PNMBackend.supports_target(cuda_target) == False

    def test_backend_initialization(self):
        """Test backend initialization."""
        from triton.backends.pnm.compiler import PNMBackend
        from triton.backends.compiler import GPUTarget

        target = GPUTarget(backend='pnm', arch='pnm_v1', warp_size=32)
        backend = PNMBackend(target)

        assert backend.binary_ext == 'pnmbin'
        assert 'pnm' in backend.hash()

    def test_parse_options(self):
        """Test option parsing."""
        from triton.backends.pnm.compiler import PNMBackend, PNMOptions
        from triton.backends.compiler import GPUTarget

        target = GPUTarget(backend='pnm', arch='pnm_v1', warp_size=32)
        backend = PNMBackend(target)

        opts = backend.parse_options({'num_compute_units': 16})
        assert isinstance(opts, PNMOptions)
        assert opts.num_compute_units == 16


class TestPNMDriver:
    """Test PNM driver functionality."""

    def test_driver_initialization(self):
        """Test driver initialization."""
        from triton.backends.pnm.driver import PNMDriver

        driver = PNMDriver()
        assert driver.utils is not None

    def test_get_current_target(self):
        """Test getting current target."""
        from triton.backends.pnm.driver import PNMDriver

        driver = PNMDriver()
        target = driver.get_current_target()

        assert target.backend == 'pnm'
        assert target.warp_size == 32

    def test_type_mapping(self):
        """Test Python to C++ type mapping."""
        from triton.backends.pnm.driver import PNMDriver

        driver = PNMDriver()

        assert driver.map_python_to_cpp_type('i32') == 'int32_t'
        assert driver.map_python_to_cpp_type('fp32') == 'float'
        assert driver.map_python_to_cpp_type('*fp16') == 'void*'


class TestPNMUtils:
    """Test PNM utility functions."""

    def test_device_properties(self):
        """Test device property queries."""
        from triton.backends.pnm.driver import PNMUtils

        utils = PNMUtils()
        props = utils.get_device_properties(0)

        assert 'compute_units' in props
        assert 'local_memory_per_cu' in props
        assert 'arch' in props

    def test_binary_loading(self):
        """Test binary loading functionality."""
        from triton.backends.pnm.driver import PNMUtils
        import struct
        import json

        utils = PNMUtils()

        # Create a test binary
        metadata = {'name': 'test_kernel'}
        meta_json = json.dumps(metadata).encode('utf-8')

        binary = (
            b'PNMB' +
            struct.pack('<I', 1) +
            struct.pack('<I', len(meta_json)) +
            meta_json +
            struct.pack('<I', 0) +
            b''
        )

        result = utils.load_binary('test_kernel', binary, 0, 0)
        assert len(result) == 5  # module, function, n_regs, n_spills, max_threads


# Integration tests (require full Triton setup)
@pytest.mark.skipif(
    os.environ.get('TRITON_PNM_INTEGRATION_TESTS', '0') != '1',
    reason="Integration tests disabled"
)
class TestPNMIntegration:
    """Integration tests with Triton."""

    def test_simple_kernel_compilation(self):
        """Test compiling a simple kernel for PNM."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        # This would compile for PNM if the backend is properly registered
        # and PNM is the active device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
