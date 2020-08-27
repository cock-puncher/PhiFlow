from abc import ABC

from phi.math.backend import Backend
from phi import math, struct
from phi.geom import assert_same_rank

from ._field import Field, SampledField
from ..math import Shape


class AnalyticField(Field, ABC):

    def unstack(self, dimension):
        components = []
        size = self.shape.get_size(dimension)
        for i in range(size):
            def _context(index=i):
                return lambda x: x.unstack()[index]
            components.append(_SymbolicOpField(_context(i), [self]))
        return components

    def _op2(self, other, operator):
        if isinstance(other, SampledField):
            self_sampled = self.at(other)
            data = operator(self_sampled.data, other.data)
            return other.with_data(data)
        new_shape = self.shape.combined(other.shape)
        return _SymbolicOpField(new_shape, operator, [self, other])

    def _op1(self, operator):
        return _SymbolicOpField(self.shape, operator, [self])


class SymbolicFieldBackend(Backend):
    # Abstract mehtods are overridden generically.
    # pylint: disable-msg = abstract-method

    def __init__(self, backend):
        Backend.__init__(self, 'Symbolic Field Backend')
        self.backend = backend
        for fname in dir(self):
            if fname not in ('__init__', 'is_tensor', 'symbolic_call', 'complex_type', 'float_type') and not fname.startswith('__'):
                function = getattr(self, fname)
                if callable(function):
                    def context(fname=fname):
                        def proxy(*args, **kwargs):
                            return self.symbolic_call(fname, args, kwargs)
                        return proxy
                    setattr(self, fname, context())

    def symbolic_call(self, func, args, kwargs):
        assert len(kwargs) == 0, 'kwargs not supported'
        backend_func = getattr(self.backend, func)
        return _SymbolicOpField(backend_func, args)

    def is_tensor(self, x, only_native=False):
        return isinstance(x, AnalyticField)


class _SymbolicOpField(AnalyticField):

    def __init__(self, shape, function, function_args):
        self.function = function
        self.function_args = function_args
        fields = filter(lambda arg: isinstance(arg, Field), function_args)
        self.fields = tuple(fields)
        self._shape = shape

    @property
    def shape(self) -> Shape:
        return self._shape

    def sample_at(self, points, reduce_channels=()) -> math.Tensor:
        args = []
        for arg in self.function_args:
            if isinstance(arg, Field):
                arg = arg.sample_at(points, reduce_channels=reduce_channels)
            args.append(arg)
        applied = self.function(*args)
        return applied

    def unstack(self, dimension):
        unstacked = {}
        for arg in self.function_args:
            if isinstance(arg, Field):
                unstacked[arg] = arg.unstack()
            elif math.is_tensor(arg) and math.ndims(arg) > 0:
                unstacked[arg] = math.unstack(arg, axis=-1, keepdims=True)
            else:
                unstacked[arg] = [arg] * self.component_count
            assert len(unstacked[arg]) == self.component_count
        result = [_SymbolicOpField(self.function, [unstacked[arg][i] for arg in self.function_args]) for i in range(self.component_count)]
        return result

