"""Multiscale solver: a coarse stage on Morton cells, then block-sparse fine updates."""

from ._preprocess import preprocess_coordinates
from .envelope import Envelope
from .solver import MultiscaleInfo, check_limits, solve_multiscale

__all__ = ["Envelope", "MultiscaleInfo", "check_limits", "preprocess_coordinates", "solve_multiscale"]
