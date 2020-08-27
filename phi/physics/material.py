from __future__ import annotations

from phi.math import extrapolation


class Material:
    """
    Defines the extrapolation modes / boundary conditions for a surface.
    The surface can be an obstacle or the domain boundary.
    """
    def __init__(self, name, grid_extrapolation, vector_extrapolation, active_extrapolation, accessible_extrapolation):
        self.name = name
        self.grid_extrapolation = grid_extrapolation
        self.vector_extrapolation = vector_extrapolation
        self.active_extrapolation = active_extrapolation
        self.accessible_extrapolation = accessible_extrapolation

    def __repr__(self):
        return self.name

    @staticmethod
    def as_material(obj: Material or tuple or list) -> Material:
        if isinstance(obj, Material):
            return obj
        else:
            grid_extrapolation = extrapolation.MixedExtrapolation()
            return Material('mix', grid_extrapolation, )


OPEN = Material('open', extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ONE)
CLOSED = NO_STICK = SLIPPERY = Material('slippery', extrapolation.BOUNDARY, extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO)
NO_SLIP = STICKY = Material('sticky', extrapolation.BOUNDARY, extrapolation.ZERO, extrapolation.ZERO, extrapolation.ZERO)
PERIODIC = Material('periodic', extrapolation.PERIODIC, extrapolation.PERIODIC, extrapolation.ONE, extrapolation.ONE)
