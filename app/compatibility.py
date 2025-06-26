"""
Utility module to handle compatibility checks and warnings for dependencies.

This module provides functions to check compatibility between required and installed
versions of libraries, and provides helpful messages for resolving issues.
"""

import importlib
import logging
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from .logger import logger

# Global flag to track if warnings have been logged
_warnings_logged = False

# Dictionary of known compatibility issues and their solutions
COMPATIBILITY_ISSUES = {
    "pyannote_version": {
        "message": "Model was trained with pyannote.audio 0.0.1, yours is {current}. "
                  "Bad things might happen unless you revert pyannote.audio to 0.x.",
        "solution": "Consider installing pyannote.audio 0.x: pip install 'pyannote.audio==0.0.1'"
    },
    "torch_version": {
        "message": "Model was trained with torch 1.10.0+cu102, yours is {current}. "
                  "Bad things might happen unless you revert torch to 1.x.",
        "solution": "Consider installing PyTorch 1.x: pip install torch==1.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html"
    },
    "cudnn_missing": {
        "message": "Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file",
        "solution": "See 'Installing CUDA Libraries' section in README.md or run 'scripts/install_cuda_libs.sh'."
    }
}

def check_library_version(library_name: str) -> Optional[str]:
    """Check the installed version of a library.
    
    Args:
        library_name: Name of the library to check
        
    Returns:
        Version string if library is installed, None otherwise
    """
    try:
        lib = importlib.import_module(library_name)
        return getattr(lib, "__version__", "unknown")
    except ImportError:
        return None

def check_cuda_libraries() -> List[str]:
    """Check for missing CUDA libraries.
    
    Returns:
        List of missing library names
    """
    cuda_libs = [
        "libcudnn_ops_infer.so.8",
        "libcudnn.so.8",
        "libcublas.so.11"
    ]
    
    missing_libs = []
    
    for lib in cuda_libs:
        try:
            # Try to find the library in common directories
            lib_paths = [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/local/cuda/lib64",
                "/usr/lib",
                "/usr/local/lib"
            ]
            
            # Add paths from the LD_LIBRARY_PATH environment variable
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            if ld_library_path:
                for path in ld_library_path.split(":"):
                    if path and path not in lib_paths:
                        lib_paths.append(path)
                        
            lib_found = False
            for path in lib_paths:
                if os.path.exists(os.path.join(path, lib)):
                    lib_found = True
                    break
                    
            if not lib_found:
                # Use ldconfig as a fallback check
                try:
                    result = subprocess.run(
                        ["ldconfig", "-p"], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    if lib not in result.stdout:
                        missing_libs.append(lib)
                    else:
                        lib_found = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    # ldconfig not available or failed
                    if not lib_found:
                        missing_libs.append(lib)
            
        except Exception:
            # If any error occurs during the check, assume the lib might be missing
            missing_libs.append(lib)
            
    return missing_libs

def check_compatibility() -> Dict[str, Dict[str, str]]:
    """Perform compatibility checks and return issues found.
    
    Returns:
        Dictionary of issues found with their details
    """
    issues = {}
    
    # Only check CUDA libraries when using Linux
    # (CUDA libraries have different filenames on Windows)
    if platform.system() == "Linux":
        # Check CUDA libraries
        missing_cuda_libs = check_cuda_libraries()
        if missing_cuda_libs:
            formatted_missing = ", ".join(missing_cuda_libs)
            issues["cudnn_missing"] = {
                "missing": missing_cuda_libs,
                "message": f"Missing CUDA libraries: {formatted_missing}",
                "solution": COMPATIBILITY_ISSUES["cudnn_missing"]["solution"]
            }
    
    # Check pyannote version
    pyannote_version = check_library_version("pyannote.audio")
    if pyannote_version and not pyannote_version.startswith("0."):
        issues["pyannote_version"] = {
            "current": pyannote_version,
            "message": COMPATIBILITY_ISSUES["pyannote_version"]["message"].format(current=pyannote_version),
            "solution": COMPATIBILITY_ISSUES["pyannote_version"]["solution"]
        }
    
    # Check torch version
    torch_version = check_library_version("torch")
    if torch_version and not torch_version.startswith("1."):
        issues["torch_version"] = {
            "current": torch_version,
            "message": COMPATIBILITY_ISSUES["torch_version"]["message"].format(current=torch_version),
            "solution": COMPATIBILITY_ISSUES["torch_version"]["solution"]
        }
    
    return issues

def generate_fallback_to_cpu_instructions() -> str:
    """Generate instructions for falling back to CPU mode if CUDA is unavailable."""
    instructions = [
        "If installing CUDA libraries is not an option, you can run in CPU mode:",
        "1. Set environment variable: export DEVICE=cpu",
        "2. Set environment variable: export COMPUTE_TYPE=int8",
        "3. Restart the application",
        "",
        "Note: CPU mode will be significantly slower than GPU acceleration."
    ]
    return "\n".join(instructions)

def log_compatibility_warnings():
    """Log compatibility warnings if issues are found."""
    global _warnings_logged
    
    # Only log warnings once per application run
    if _warnings_logged:
        return
        
    issues = check_compatibility()
    
    if not issues:
        return
    
    # Mark warnings as logged
    _warnings_logged = True
    
    logger.warning("=" * 80)
    logger.warning("COMPATIBILITY WARNINGS DETECTED")
    logger.warning("=" * 80)
    
    for issue_type, details in issues.items():
        logger.warning(details["message"])
        logger.warning("SOLUTION: %s", details["solution"])
        
        # Add special handling for CUDA library issues
        if issue_type == "cudnn_missing":
            logger.warning("ALTERNATIVE: %s", generate_fallback_to_cpu_instructions())
    
    logger.warning("=" * 80)
