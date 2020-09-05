from __future__ import annotations

from .backend import math as native_math
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
        raise NotImplementedError()

    def transform_coordinates(self, coordinates, shape):
        return NotImplemented

    def evaluate(self, value: Tensor, coordinates):
        raise NotImplementedError()


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


class _StatelessExtrapolation(Extrapolation):

    def to_dict(self) -> dict:
        return {'type': repr(self)}

    def gradient(self):
        raise NotImplementedError()

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
PERIODIC = _PeriodicExtrapolation(1)
BOUNDARY = _BoundaryExtrapolation(2)
SYMMETRIC = _SymmetricExtrapolation(3)
REFLECT = _ReflectExtrapolation(4)


class MixedExtrapolation(Extrapolation):

    def __init__(self, lower_upper_by_axis: dict):
        Extrapolation.__init__(self, None)
        self.ext = {ax: (e, e) if isinstance(e, Extrapolation) else tuple(e) for ax, e in lower_upper_by_axis.items()}

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


