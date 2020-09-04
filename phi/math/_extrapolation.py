from __future__ import annotations

from .backend import math as native_math
from ._tensors import Tensor, NativeTensor, CollapsedTensor, TensorStack, tensor
from . import _tensor_math as math


class IncompatibleExtrapolations(ValueError):
    def __init__(self, extrapolation1, extrapolation2):
        ValueError.__init__(self, extrapolation1, extrapolation2)


class Extrapolation:

    def to_dict(self) -> dict:
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
        raise NotImplementedError()

    def transform_coordinates(self, coordinates, shape):
        return NotImplemented

    def evaluate(self, value: Tensor, coordinates):
        raise NotImplementedError()


class ConstantExtrapolation(Extrapolation):

    def __init__(self, value):
        self.value = tensor(value)

    def __repr__(self):
        return repr(self.value)

    def to_dict(self) -> dict:
        return {'type': 'constant', 'value': self.value.numpy()}

    def gradient(self):
        return ZERO

    def pad(self, value: Tensor, widths: dict):
        value = tensor(value)
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

    def __eq__(self, other):
        return isinstance(other, ConstantExtrapolation) and math.close(self.value, other.value)

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


class _StatelessExtrapolation(Extrapolation):

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    def gradient(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return type(other) == type(self)

    def _op(self, other, op):
        if type(other) == type(self):
            return self
        elif isinstance(other, Extrapolation) and not isinstance(other, _StatelessExtrapolation):
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


class _BoundaryExtrapolation(_StatelessExtrapolation):
    """
    Uses the closest defined value for points lying outside the defined region.
    """
    def __repr__(self):
        return 'boundary'

    def gradient(self):
        return ZERO


class _PeriodicExtrapolation(_StatelessExtrapolation):
    def __repr__(self):
        return 'periodic'

    def gradient(self):
        return self


class _SymmetricExtrapolation(_StatelessExtrapolation):
    """
    Mirror with the boundary value occurring twice.
    """
    def __repr__(self):
        return 'symmetric'

    def gradient(self):
        return -self


class _ReflectExtrapolation(_StatelessExtrapolation):
    """
    Mirror of inner elements. The boundary value is not duplicated.
    """
    def __repr__(self):
        return 'reflect'

    def gradient(self):
        return -self


ZERO = ConstantExtrapolation(0)
ONE = ConstantExtrapolation(1)
PERIODIC = _PeriodicExtrapolation()
BOUNDARY = _BoundaryExtrapolation()
SYMMETRIC = _SymmetricExtrapolation()
REFLECT = _ReflectExtrapolation()


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


