from phi import math
from ._geom import Geometry
from ._empty import NO_GEOMETRY
from ._transform import rotate
from ._box import bounding_box, AABox
from ..math import combined_shape


class Union(Geometry):

    def __init__(self, geometries):
        self._geometries = tuple(geometries)
        assert len(self._geometries) > 0
        for g in self._geometries[1:]:
            assert g.rank == self._geometries[0]
        self._shape = combined_shape(*[g.shape for g in geometries])

    @property
    def shape(self):
        return self._shape

    @property
    def geometries(self):
        return self._geometries

    @property
    def rank(self):
        return self.geometries[0].rank

    def lies_inside(self, location):
        return math.any([geometry.lies_inside(location) for geometry in self.geometries], axis=0)

    def approximate_signed_distance(self, location):
        return math.min([geometry.approximate_signed_distance(location) for geometry in self.geometries], axis=0)

    @property
    def center(self):
        return self._bounding_box().center

    def bounding_radius(self):
        return self._bounding_box().bounding_radius()

    def bounding_half_extent(self):
        return self._bounding_box().bounding_half_extent()

    def _bounding_box(self):
        boxes = [bounding_box(g) for g in self.geometries]
        lower = math.min([b.lower for b in boxes], axis=0)
        upper = math.max([b.upper for b in boxes], axis=0)
        return AABox(lower, upper)

    def shifted(self, delta):
        return Union([geometry.shifted(delta) for geometry in self.geometries])

    def rotated(self, angle):
        return rotate(self, angle)


def union(*geometries) -> Geometry:
    """
    Union of the given geometries.
    A point lies inside the union if it lies within at least one of the geometries.

    :param geometries: arbitrary geometries with same spatial dimensions. Arbitrary batch dimensions are allowed.
    :return: union Geometry
    """
    if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)):
        geometries = geometries[0]
    if len(geometries) == 0:
        return NO_GEOMETRY
    elif len(geometries):
        return geometries[0]
    else:
        base_geometries = ()
        for geometry in geometries:
            base_geometries += geometry.geometries if isinstance(geometry, Union) else (geometry,)
        return Union(base_geometries)


Geometry.__add__ = lambda g1, g2: union(g1, g2)
