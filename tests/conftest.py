"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model_path": "/tmp/test_model",
        "source_path": "/tmp/test_data",
        "iterations": 100,
        "test_iterations": [50, 100],
        "save_iterations": [50, 100],
        "checkpoint_iterations": [],
        "resolution": 1,
        "white_background": False,
        "data_device": "cpu",
        "eval": False,
        "sh_degree": 3,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "densification_interval": 100,
        "opacity_reset_interval": 3000,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densify_grad_threshold": 0.0002,
    }


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """Generate a sample 3D point cloud for testing."""
    num_points = 100
    points = np.random.randn(num_points, 3).astype(np.float32)
    return points


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """Generate sample image data for testing."""
    height, width = 64, 64
    image = np.random.rand(height, width, 3).astype(np.float32)
    return image


@pytest.fixture
def mock_camera_params() -> Dict[str, Any]:
    """Provide mock camera parameters for testing."""
    return {
        "width": 800,
        "height": 600,
        "fovx": 1.0472,  # 60 degrees in radians
        "fovy": 0.7854,  # 45 degrees in radians
        "znear": 0.01,
        "zfar": 100.0,
        "world_view_transform": np.eye(4, dtype=np.float32),
        "full_proj_transform": np.eye(4, dtype=np.float32),
        "camera_center": np.array([0, 0, 0], dtype=np.float32),
    }


@pytest.fixture
def temp_ply_file(temp_dir: Path, sample_point_cloud: np.ndarray) -> Path:
    """Create a temporary PLY file with sample point cloud data."""
    ply_path = temp_dir / "test_points.ply"
    
    # Simple PLY file writing (minimal implementation for testing)
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(sample_point_cloud)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in sample_point_cloud:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    return ply_path


@pytest.fixture
def mock_training_args():
    """Mock training arguments for testing."""
    class Args:
        def __init__(self):
            self.source_path = "/tmp/test_data"
            self.model_path = "/tmp/test_model"
            self.iterations = 1000
            self.test_iterations = [500, 1000]
            self.save_iterations = [1000]
            self.checkpoint_iterations = []
            self.quiet = True
            self.resolution = 1
            self.white_background = False
            self.data_device = "cpu"
            self.eval = False
    
    return Args()


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    
    def _capture():
        monkeypatch.setattr(sys, 'stdout', captured_output)
        return captured_output
    
    return _capture


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seeds for reproducible tests."""
    np.random.seed(42)
    
    # If using PyTorch
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture
def mock_gaussian_model_data() -> Dict[str, np.ndarray]:
    """Provide mock data for Gaussian model testing."""
    num_gaussians = 50
    return {
        "means": np.random.randn(num_gaussians, 3).astype(np.float32),
        "scales": np.random.rand(num_gaussians, 3).astype(np.float32) * 0.1,
        "rotations": np.random.randn(num_gaussians, 4).astype(np.float32),
        "opacities": np.random.rand(num_gaussians, 1).astype(np.float32),
        "features": np.random.rand(num_gaussians, 48).astype(np.float32),
    }


@pytest.fixture
def environment_setup(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    # Cleanup happens automatically with monkeypatch