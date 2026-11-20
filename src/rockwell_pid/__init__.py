"""Positional and velocity form PID controllers, implementing the
equations in Rockwell Automation's white paper "Perform Common Process
Loop Control Algorithms Using the PIDE Instruction" (Logix-WP008),
page 2, for comparison with the reference PID controller of Sundström
et al. (2024) in python_pid.

These are direct implementations of the two equations in the paper,
not of the PIDE instruction itself (which is a velocity form with more
features)."""

from .positional import PositionalPID
from .velocity import VelocityPID

__all__ = ["PositionalPID", "VelocityPID"]
