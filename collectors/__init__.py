"""
AIO Collectors Package

This package provides data collection functionality for different benchmarks:
- IOR: I/O performance benchmark
- HACC: Cosmology simulation I/O benchmark
"""

from .ior_collector import IORCollector
from .hacc_collector import HACCCollector
from .base_collector import BaseCollector

__all__ = ['IORCollector', 'HACCCollector', 'BaseCollector']
