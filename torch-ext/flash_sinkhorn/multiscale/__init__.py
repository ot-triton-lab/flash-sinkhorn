"""Multiscale solver: a coarse stage on Morton cells, then block-sparse fine updates."""

from ._preprocess import preprocess_coordinates
from .solver import MultiscaleInfo, solve_multiscale

__all__ = ["MultiscaleInfo", "preprocess_coordinates", "solve_multiscale"]
