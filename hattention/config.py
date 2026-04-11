"""
Configuration utilities for hattention module.
Handles dynamic path resolution for FLA (Flash-Linear-Attention) dependencies.
"""

import os
from pathlib import Path


def get_fla_base_path():
    """
    Resolve the FLA (Flash-Linear-Attention) base path dynamically.
    
    Priority:
    1. Environment variable FLA_BASE_PATH (if set)
    2. Auto-detect: {project_root}/flame/3rdparty/flash-linear-attention/
    
    Returns:
        str: Path to FLA base directory (with trailing slash)
        
    Raises:
        RuntimeError: If path cannot be found
    """
    # Check env var first
    if "FLA_BASE_PATH" in os.environ:
        path = os.environ["FLA_BASE_PATH"]
        if not os.path.isdir(path):
            raise RuntimeError(f"FLA_BASE_PATH set but directory not found: {path}")
        return path.rstrip("/") + "/"
    
    # Auto-detect relative to hattention module
    hattention_dir = Path(__file__).parent
    project_root = hattention_dir.parent
    fla_path = project_root / "flame" / "3rdparty" / "flash-linear-attention"
    
    if fla_path.is_dir():
        return str(fla_path) + "/"
    
    raise RuntimeError(
        f"Cannot find FLA base path. Tried: {fla_path}\n"
        f"Solutions:\n"
        f"  1. Ensure 'git submodule update --init --recursive' was run\n"
        f"  2. Set FLA_BASE_PATH environment variable: "
        f"export FLA_BASE_PATH=/path/to/flash-linear-attention/"
    )
