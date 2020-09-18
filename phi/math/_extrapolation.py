from __future__ import annotations

from abc import ABC

from .backend import math as native_math
from ._shape import Shape
from ._tensors import Tensor, NativeTensor, CollapsedTensor, TensorStack, tensor
from . import _tensor_math as math


class IncompatibleExtrapolations(ValueError):
    def __init__(self, extrapolation1, extrapolation2):
        ValueError.__init__(self, extrapolation1, extrapolation2)


class Extrapolation:

    def __init__(self, pad_rank):
        """
        Extrapolations are used to determine values of grids or other structures outside the sampled bounds.

        They play a pivotal role in padding and sampling.

        :param pad_rank: low-ranking extrapolations are handled first during mixed-extrapolation padding.
        The typical order is periodic=1, boundary=2, symmetric=3, reflect=4, constant=5.
        """
        self.pad_rank = pad_rank

    def to_dict(self) -> dict:
        """
        Serialize this extrapolation to a dictionary that is JSON-writable.

        Use extrapolation.from_dict() to restore the Extrapolation object.
        """
        raise NotImplementedError()

    def gradient(self) -> Extrapolation:
        """
        Returns the extrapolation for the spatial gradient of a tensor/field with this extrapolation.

        :rtype: _Extrapolation
        """
        raise NotImplementedError()

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        """
        Pads a tensor using this extrapolation.

        :param value: tensor to be padded
        :param widths: name: str -> (lower: int, upper: int)
        """
        for dim in widths:
            left_pad_values = self.pad_values(value, widths[dim][False], dim, False)
            right_pad_values = self.pad_values(value, widths[dim][True], dim, True)
            value = math._stack([left_pad_values, value, right_pad_values], dim, value.shape.get_type(dim))
        return value

    def pad_values(self, value: Tensor, width: int, dimension: str, upper_edge: bool) -> Tensor:
        """
        Determines the values with which the given tensor would be padded at the specified using this extrapolation.

        :param value: tensor to be padded
        :param width: number of cells to pad perpendicular to the face
        :param dimension: axis in which to pad
        :param upper_edge: True for upper edge, False for lower edge
        :return: tensor that can be concatenated to value for padding
        """
        raise NotImplementedError()

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        """
        If is_copy_pad, transforms outsider coordinates to point to the index from which the value should be copied.
        Otherwise, returns NotImplemented.

        :param coordinates: integer coordinates in index space
        :param shape: tensor shape
        :return: transformed coordinates or NotImplemented
        """
        return NotImplemented

    @property
    def is_copy_pad(self):
        """
        :return: True if all pad values are copies of existing values in the tensor to be padded
        """
        return False

    def __getitem__(self, item):
        return self


class ConstantExtrapolation(Extrapolation):

    def __init__(self, value):
        Extrapolation.__init__(self, 5)
        self.value = tensor(value)

    def __repr__(self):
        return repr(self.value)

    def to_dict(self) -> dict:
        return {'type': 'constant', 'value': self.value.numpy()}

    def gradient(self):
        return ZERO

    def pad(self, value: Tensor, widths: dict):
        if isinstance(value, NativeTensor):
            native = value.tensor
            ordered_pad_widths = value.shape.order(widths, default=(0, 0))
            result_tensor = native_math.pad(native, ordered_pad_widths, 'constant', self.value)
            new_shape = value.shape.with_sizes(native_math.staticshape(result_tensor))
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            if value.tensor.shape.volume > 1 or not math.close(self.value, value.tensor):
                return self.pad(value.expand(), widths)
            else:  # Stays constant value, only extend shape
                new_sizes = []
                for size, dim, dim_type in value.shape.dimensions:
                    if dim not in widths:
                        new_sizes.append(size)
                    else:
                        delta = sum(widths[dim]) if isinstance(widths[dim], (tuple, list)) else 2 * widths[dim]
                        new_sizes.append(size + int(delta))
                new_shape = value.shape.with_sizes(new_sizes)
                return CollapsedTensor(value.tensor, new_shape)
        # elif isinstance(value, SparseLinearOperation):
        #     return pad_operator(value, pad_width, mode)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._cache(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value.stack_dim_name}
            tensors = [self.pad(t, inner_widths) for t in value.tensors]
            return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type, value.keep_separate)
        else:
            raise NotImplementedError()

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, ConstantExtrapolation) and math.close(self.value, other.value)

    def __hash__(self):
        return hash(self.__class__)

    def is_zero(self):
        return self == ZERO

    def is_one(self):
        return self == ONE

    def __add__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value + other.value)
        elif self.is_zero():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __sub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value - other.value)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __rsub__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value - self.value)
        elif self.is_zero():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __mul__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value * other.value)
        elif self.is_one():
            return other
        elif self.is_zero():
            return self
        else:
            raise IncompatibleExtrapolations(self, other)

    def __truediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value / other.value)
        elif self.is_zero():
            return self
        else:
            raise IncompatibleExtrapolations(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(other.value / self.value)
        elif self.is_one():
            return other
        else:
            raise IncompatibleExtrapolations(self, other)

    def __lt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value < other.value)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __gt__(self, other):
        if isinstance(other, ConstantExtrapolation):
            return ConstantExtrapolation(self.value > other.value)
        else:
            raise IncompatibleExtrapolations(self, other)


class _CopyExtrapolation(Extrapolation, ABC):

    @property
    def is_copy_pad(self):
        return True

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        if isinstance(value, NativeTensor):
            native = value.tensor
            ordered_pad_widths = value.shape.order(widths, default=(0, 0))
            result_tensor = native_math.pad(native, ordered_pad_widths, repr(self))
            new_shape = value.shape.with_sizes(math.staticshape(result_tensor))
            return NativeTensor(result_tensor, new_shape)
        elif isinstance(value, CollapsedTensor):
            inner = value.tensor
            inner_widths = {dim: w for dim, w in widths.items() if dim in inner.shape}
            if len(inner_widths) > 0:
                inner = self.pad(inner, widths)
            new_sizes = []
            for size, dim, dim_type in value.shape.dimensions:
                if dim not in widths:
                    new_sizes.append(size)
                else:
                    delta = sum(widths[dim]) if isinstance(widths[dim], (tuple, list)) else 2 * widths[dim]
                    new_sizes.append(size + int(delta))
            new_shape = value.shape.with_sizes(new_sizes)
            return CollapsedTensor(inner, new_shape)
        # elif isinstance(value, SparseLinearOperation):
        #     return pad_operator(value, widths, mode)
        elif isinstance(value, TensorStack):
            if not value.requires_broadcast:
                return self.pad(value._cache(), widths)
            inner_widths = {dim: w for dim, w in widths.items() if dim != value.stack_dim_name}
            tensors = [self.pad(t, inner_widths) for t in value.tensors]
            return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type, value.keep_separate)
        else:
            raise NotImplementedError()

    def __eq__(self, other):
        return type(other) == type(self)

    def __hash__(self):
        return hash(self.__class__)

    def _op(self, other, op):
        if type(other) == type(self):
            return self
        elif isinstance(other, Extrapolation) and not isinstance(other, _CopyExtrapolation):
            op = getattr(other, op.__name__)
            return op(self)
        else:
            raise IncompatibleExtrapolations(self, other)

    def __add__(self, other):
        return self._op(other, ConstantExtrapolation.__add__)

    def __mul__(self, other):
        return self._op(other, ConstantExtrapolation.__mul__)

    def __sub__(self, other):
        return self._op(other, ConstantExtrapolation.__rsub__)

    def __truediv__(self, other):
        return self._op(other, ConstantExtrapolation.__rtruediv__)

    def __lt__(self, other):
        return self._op(other, ConstantExtrapolation.__gt__)

    def __gt__(self, other):
        return self._op(other, ConstantExtrapolation.__lt__)


class _BoundaryExtrapolation(_CopyExtrapolation):
    """
    Uses the closest defined value for points lying outside the defined region.
    """
    def __repr__(self):
        return 'boundary'

    def gradient(self):
        return ZERO

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        return math.clip(coordinates, 0, math.tensor(shape - 1))

    def concat_pad(self, value: Tensor, widths: dict) -> Tensor:
        raise NotImplementedError()
        dims = range(math.ndims(value))
        for dim in dims:
            pad_lower, pad_upper = pad_width[dim]
            if pad_lower == 0 and pad_upper == 0:
                continue  # Nothing to pad
            bottom_row = value[
                (slice(None),) + tuple([slice(1) if d == dim else slice(None) for d in dims]) + (slice(None),)]
            top_row = value[
                (slice(None),) + tuple([slice(-1, None) if d == dim else slice(None) for d in dims]) + (slice(None),)]
            value = math.concat([bottom_row] * pad_lower + [value] + [top_row] * pad_upper)
        return value


class _PeriodicExtrapolation(_CopyExtrapolation):
    def __repr__(self):
        return 'periodic'

    def gradient(self):
        return self

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        return coordinates % shape

    def concat_pad(self, value: Tensor, widths: dict) -> Tensor:
        raise NotImplementedError()
        dims = range(math.ndims(value))
        for dim in dims:
            pad_lower, pad_upper = pad_width[dim]
            if pad_lower == 0 and pad_upper == 0:
                continue  # Nothing to pad
            lower = value[tuple([slice(value.shape[dim] - pad_lower, None) if d == dim else slice(None) for d in dims])]
            upper = value[tuple([slice(None, pad_upper) if d == dim else slice(None) for d in dims])]
            value = math.concat([lower, value, upper], axis=dim)
        return value


class _SymmetricExtrapolation(_CopyExtrapolation):
    """
    Mirror with the boundary value occurring twice.
    """
    def __repr__(self):
        return 'symmetric'

    def gradient(self):
        return -self

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = coordinates % (2 * shape)
        return ((2 * shape - 1) - abs((2 * shape - 1) - 2 * coordinates)) // 2

    def concat_pad(self, value: Tensor, widths: dict) -> Tensor:
        raise NotImplementedError()
        raise NotImplementedError()  # only used by PyTorch which does not support ::-1 axis flips
        dims = range(math.ndims(value))
        for dim in dims:
            pad_lower, pad_upper = pad_width[dim]
            if pad_lower == 0 and pad_upper == 0:
                continue  # Nothing to pad
            top_rows = value[
                tuple([slice(value.shape[dim] - pad_upper, None) if d == dim else slice(None) for d in dims])]
            bottom_rows = value[tuple([slice(None, pad_lower) if d == dim else slice(None) for d in dims])]
            top_rows = math.flip_axis(top_rows, dim)
            bottom_rows = math.flip_axis(bottom_rows, dim)
            value = math.concat([bottom_rows, value, top_rows], axis=dim)
        return value


class _ReflectExtrapolation(_CopyExtrapolation):
    """
    Mirror of inner elements. The boundary value is not duplicated.
    """
    def __repr__(self):
        return 'reflect'

    def gradient(self):
        return -self

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = coordinates % (2 * shape - 2)
        return (shape - 1) - math.abs((shape - 1) - coordinates)


ZERO = ConstantExtrapolation(0)
ONE = ConstantExtrapolation(1)
PERIODIC = _PeriodicExtrapolation(1)
BOUNDARY = _BoundaryExtrapolation(2)
SYMMETRIC = _SymmetricExtrapolation(3)
REFLECT = _ReflectExtrapolation(4)


class MixedExtrapolation(Extrapolation):

    def __init__(self, extrapolations: dict):
        """
        A mixed extrapolation uses different extrapolations for different sides.

        :param extrapolations: axis: str -> (lower: Extrapolation, upper: Extrapolation) or Extrapolation
        """
        Extrapolation.__init__(self, None)
        self.ext = {ax: (e, e) if isinstance(e, Extrapolation) else tuple(e) for ax, e in extrapolations.items()}

    def to_dict(self) -> dict:
        return {
            'type': 'mixed',
            'axes': {ax: (es[0].to_dict(), es[1].to_dict()) for ax, es in self.ext.items()}
        }

    def gradient(self) -> Extrapolation:
        return MixedExtrapolation({ax: (es[0].gradient(), es[1].gradient()) for ax, es in self.ext.items()})

    def pad(self, value: Tensor, widths: dict) -> Tensor:
        extrapolations = set(sum(self.ext.values(), ()))
        extrapolations = tuple(sorted(extrapolations, key=lambda e: e.pad_rank))
        for ext in extrapolations:
            ext_widths = {ax: (l if self.ext[ax][0] == ext else 0, u if self.ext[ax][1] == ext else 0) for ax, (l, u) in widths.items()}
            value = ext.pad(value, ext_widths)
        return value

    def evaluate(self, value: Tensor, coordinates):
        pass

    def transform_coordinates(self, coordinates: Tensor, shape: Shape) -> Tensor:
        coordinates = math.unstack(coordinates, axis=-1)
        assert len(self.ext) == len(shape) == len(coordinates)
        result = []
        for dim, dim_coords in zip(shape.spatial.unstack(), coordinates):
            dim_extrapolations = self.ext[dim.name]
            if dim_extrapolations[0] == dim_extrapolations[1]:
                result.append(dim_extrapolations[0].transform_coordinates(dim_coords, dim))
            else:  # separate boundary for lower and upper face
                lower = dim_extrapolations[0].transform_coordinates(dim_coords, dim)
                upper = dim_extrapolations[1].transform_coordinates(dim_coords, dim)
                result.append(math.where(dim_coords <= 0, lower, upper))
        return math.channel_stack(result, 'vector')

    def __getitem__(self, item):
        dim, face = item
        return self.ext[dim][face]


def from_dict(dictionary: dict) -> Extrapolation:
    etype = dictionary['type']
    if etype == 'constant':
        return ConstantExtrapolation(dictionary['value'])
    elif etype == 'periodic':
        return PERIODIC
    elif etype == 'boundary':
        return BOUNDARY
    elif etype == 'symmetric':
        return SYMMETRIC
    elif etype == 'reflect':
        return REFLECT
    else:
        raise ValueError(dictionary)


