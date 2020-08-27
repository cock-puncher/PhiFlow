from __future__ import annotations

from abc import ABC

from phi import math
from phi.geom import Geometry
from phi.math import Shape, Tensor
from phi.math.backend import Extrapolation


class Field:

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties

        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        raise NotImplementedError()

    @property
    def rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.shape.spatial.rank

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        """
        Sample this field at the world-space locations (in physical units) given by points.

        Points must be one of the following:

        * **Tensor** with exactly one channel dimension.
          The channel dimension holds the vectors that reference the locations in world-space.
          Batch dimensions are matched with the batch dimensions of this Field.
          Spatial dimensions can be used to sample a grid of locations.

        * **Geometry**. Approximates the mean field value inside the volume.
          For small volumes, the value at the volume's center may be sampled.
          The batch dimensions of the geometry are matched with this Field.
          Spatial dimensions can be used to sample a grid of geometries.

        * **List** or **tuple** of any of these. This broadcasts the sampling for all entries in the list.
          The result will have the same (nested) structure.

        :param points: world-space locations
        :param reduce_channels: batch dimensions to be reduced against channel dimensions. Indicates that the different channels of this field should be sampled at different locations.
        :return: object of same kind as points
        """
        # * **Field**. The values of that field are interpreted as the sample locations. Analytic fields cannot be used.
        raise NotImplementedError(self)

    def at(self, representation: SampledField) -> SampledField:
        elements = representation.elements
        resampled = self.sample_at(elements, reduce_channels=elements.shape.non_channel.without(representation.shape).names)
        extrap = self.extrapolation if isinstance(self, SampledField) else representation.extrapolation
        return representation._op1(lambda old: extrap if isinstance(old, math.extrapolation.Extrapolation) else resampled)

    def unstack(self, dimension: str) -> tuple:
        """
        Unstack the field along one of its dimensions.
        The dimension can be batch, spatial or channel.

        :param dimension: name of the dimension to unstack, must be part of `self.shape`
        :return: tuple of Fields
        """
        raise NotImplementedError()

    def __mul__(self, other):
        return self._op2(other, lambda d1, d2: d1 * d2)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op2(other, lambda d1, d2: d1 / d2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda d1, d2: d2 / d1)

    def __sub__(self, other):
        return self._op2(other, lambda d1, d2: d1 - d2)

    def __rsub__(self, other):
        return self._op2(other, lambda d1, d2: d2 - d1)

    def __add__(self, other):
        return self._op2(other, lambda d1, d2: d1 + d2)

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self._op2(power, lambda f, p: f ** p)

    def __neg__(self):
        return self._op1(lambda x: -x)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y)

    def __abs__(self):
        return self._op1(lambda x: abs(x))

    def _op1(self, operator) -> Field:
        """
        Perform an operation on the data of this field.

        :param operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions
        :return: Field of same type
        """
        raise NotImplementedError()

    def _op2(self, other, operator) -> Field:
        raise NotImplementedError()


class SampledField(Field, ABC):

    @property
    def elements(self) -> Geometry:
        """
        Returns a geometrical representation of the discretized volume elements.
        The result is a tuple of Geometry objects, each of which can have additional spatial (but not batch) dimensions.

        For grids, the geometries are boxes while particle fields may be represented as spheres.

        If this Field has no discrete points, this method returns an empty geometry.

        :return: Geometry with all batch/spatial dimensions of this Field. Staggered sample points are modelled using extra batch dimensions.
        """
        raise NotImplementedError(self)

    @property
    def points(self) -> Tensor:
        return self.elements.center

    @property
    def data(self) -> Tensor:
        raise NotImplementedError()

    def with_data(self, data: Tensor):
        raise NotImplementedError()

    @property
    def extrapolation(self) -> Extrapolation:
        raise NotImplementedError()


class IncompatibleFieldTypes(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)
