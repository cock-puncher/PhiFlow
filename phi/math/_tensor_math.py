import numbers
import re
import time
import warnings
from functools import partial

import numpy as np

from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, define_shape
from . import _extrapolation as extrapolation
from ._track import as_sparse_linear_operation, SparseLinearOperation, pad_operator, sum_operators
from .backend import math
from ._tensors import Tensor, tensor, broadcastable_native_tensors, NativeTensor, CollapsedTensor, TensorStack, combined_shape
from phi.math.backend._scipy_backend import SCIPY_BACKEND
from ._config import GLOBAL_AXIS_ORDER


def is_tensor(x):
    return isinstance(x, Tensor)


def as_tensor(x, convert_external=True):
    if convert_external:
        return tensor(x)
    else:
        return x


def copy(tensor, only_mutable=False):
    raise NotImplementedError()


def print_(value):
    """
    Print a tensor with no more than two spatial dimensions, splitting it along all batch and channel dimensions.

    Unlike regular printing, the primary axis, typically x, is oriented to the right.

    :param value: tensor-like
    """
    value = tensor(value)
    axis_order = value.shape.spatial.names if GLOBAL_AXIS_ORDER.x_last() else tuple(reversed(value.shape.spatial.names))
    if value.shape.spatial.rank == 1:
        for index_dict in value.shape.non_spatial.meshgrid():
            print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(axis_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', text))
    elif value.shape.spatial.rank == 2:
        for index_dict in value.shape.non_spatial.meshgrid():
            print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(axis_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', re.sub('\],', '', text)))
    else:
        raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')


def transpose(tensor, axes):
    if isinstance(tensor, Tensor):
        return CollapsedTensor(tensor, tensor.shape[axes])
    else:
        return math.transpose(tensor, axes)


def zeros(channels=(), batch=None, dtype=None, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: size}
    :param dtype:
    :param spatial:
    :return:
    """
    shape = define_shape(channels, batch=batch, infer_types_if_not_given=True, **spatial)
    native_zero = math.zeros((), dtype=dtype)
    collapsed = NativeTensor(native_zero, EMPTY_SHAPE)
    return CollapsedTensor(collapsed, shape)


def ones(channels=(), batch=None, dtype=None, **spatial):
    """

    :param channels: int or (int,)
    :param batch: int or {name: size}
    :param dtype:
    :param spatial:
    :return:
    """
    shape = define_shape(channels, batch, infer_types_if_not_given=True, **spatial)
    native_zero = math.ones((), dtype=dtype)
    collapsed = NativeTensor(native_zero, EMPTY_SHAPE)
    return CollapsedTensor(collapsed, shape)


def random_normal(channels=(), batch=None, dtype=None, **spatial):
    shape = define_shape(channels, batch, infer_types_if_not_given=True, **spatial)
    native = math.random_normal(shape.sizes)
    native = native if dtype is None else native.astype(dtype)
    return NativeTensor(native, shape)


def fftfreq(resolution, dtype=None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    :param resolution: grid resolution measured in cells
    :param dtype: data type of the returned tensor
    :return: tensor holding the frequencies of the corresponding values computed by math.fft
    """
    resolution = spatial_shape(resolution)
    k = math.meshgrid(*[np.fft.fftfreq(int(n)) for n in resolution.sizes])
    k = [math.to_float(channel) if dtype is None else math.cast(channel, dtype) for channel in k]
    channel_shape = spatial_shape(k[0].shape)
    k = [NativeTensor(channel, channel_shape) for channel in k]
    return TensorStack(k, 'vector', CHANNEL_DIM)


def meshgrid(*coordinates):
    indices_list = math.meshgrid(*coordinates)
    single_shape = spatial_shape([len(coo) for coo in coordinates])
    channels = [NativeTensor(t, single_shape) for t in indices_list]
    return TensorStack(channels, 'vector', CHANNEL_DIM)


def channel_stack(values, axis: str):
    return _stack(values, axis, CHANNEL_DIM)


def batch_stack(values, axis: str = 'batch'):
    return _stack(values, axis, BATCH_DIM)


def spatial_stack(values, axis: str):
    return _stack(values, axis, SPATIAL_DIM)


def _stack(values, dim: str, dim_type: int):
    assert isinstance(dim, str)
    def inner_stack(*values):
        varying_shapes = any([v.shape != values[0].shape for v in values[1:]])
        tracking = any([isinstance(v, SparseLinearOperation) for v in values])
        inner_keep_separate = any([isinstance(v, TensorStack) and v.keep_separate for v in values])
        return TensorStack(values, dim, dim_type, keep_separate=varying_shapes or tracking or inner_keep_separate)

    result = broadcast_op(inner_stack, values)
    return result


def concat(values, axis):
    tensors = broadcastable_native_tensors(values)
    concatenated = math.concat(tensors, axis)
    return NativeTensor(concatenated, values[0].shape)


def spatial_pad(value, pad_width: tuple or list, mode: 'extrapolation.Extrapolation'):
    value = tensor(value)
    return pad(value, {n: w for n, w in zip(value.shape.spatial.names, pad_width)}, mode=mode)


def pad(value: Tensor, widths: dict, mode: 'extrapolation.Extrapolation'):
    """

    :param value: tensor to be padded
    :param widths: name: str -> (lower: int, upper: int)
    :param mode: extrapolation object
    :return:
    """
    return mode.pad(value, widths)

    value = tensor(value)
    if isinstance(value, NativeTensor):
        native = value.tensor
        ordered_pad_widths = value.shape.order(pad_width, default=0)
        ordered_mode = value.shape.order(mode, default=extrapolation.ZERO)
        result_tensor = math.pad(native, ordered_pad_widths, ordered_mode)
        new_shape = value.shape.with_sizes(math.staticshape(result_tensor))
        return NativeTensor(result_tensor, new_shape)
    elif isinstance(value, CollapsedTensor):
        inner = value.tensor
        inner_widths = {dim: w for dim, w in pad_width.items() if dim in inner.shape}
        if len(inner_widths) > 0:
            inner = pad(inner, pad_width, mode=mode)
        new_sizes = []
        for size, dim, dim_type in value.shape.dimensions:
            if dim not in pad_width:
                new_sizes.append(size)
            else:
                delta = sum_(pad_width[dim]) if isinstance(pad_width[dim], (tuple, list)) else 2 * pad_width[dim]
                new_sizes.append(size + int(delta))
        new_shape = value.shape.with_sizes(new_sizes)
        return CollapsedTensor(inner, new_shape)
    elif isinstance(value, SparseLinearOperation):
        return pad_operator(value, pad_width, mode)
    elif isinstance(value, TensorStack):
        if not value.requires_broadcast:
            return pad(value._cache())
        inner_widths = {dim: w for dim, w in pad_width.items() if dim != value.stack_dim_name}
        tensors = [pad(t, inner_widths, mode) for t in value.tensors]
        return TensorStack(tensors, value.stack_dim_name, value.stack_dim_type, value.keep_separate)
    else:
        raise NotImplementedError()


def resample(inputs, sample_coords, boundary):
    if isinstance(boundary, (tuple, list)):
        boundary = [extrapolation.ZERO, *boundary, extrapolation.ZERO]

    def atomic_resample(inputs, sample_coords):
        inputs_, _ = _invertible_standard_form(inputs)
        sample_coords_, _ = _invertible_standard_form(sample_coords)  # TODO iterate if keep_separate
        resampled = math.resample(inputs_, sample_coords_, 'linear', boundary)

        batch_shape = inputs.shape.batch & sample_coords.shape.batch
        result_shape = batch_shape & sample_coords.shape.spatial & inputs.shape.channel

        un_reshaped = math.reshape(resampled, result_shape.sizes)
        return NativeTensor(un_reshaped, result_shape)

    result = broadcast_op(atomic_resample, [inputs, sample_coords])
    return result


def closest_grid_values(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    """

    :param extrap: grid extrapolation
    :param grid: grid data. The grid is spanned by the spatial dimensions of the tensor
    :param coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
    :return: Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)
    """
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather_nd.
    assert all(name not in grid.shape for name in coordinates.shape.spatial.names), 'grid and coordinates must have different spatial dimensions'
    # --- Compute weights ---
    sp_rank = grid.shape.spatial.rank
    min_coords = to_int(floor(coordinates))
    max_coords = extrap.transform_coordinates(min_coords + 1, grid.shape)
    min_coords = extrap.transform_coordinates(min_coords, grid.shape)

    def interpolate_nd(is_hi_by_axis, axis):
        is_hi_by_axis_2 = is_hi_by_axis | np.array([ax == axis for ax in range(sp_rank)])
        coords_left = math.where(is_hi_by_axis, max_coords, min_coords)
        coords_right = math.where(is_hi_by_axis_2, max_coords, min_coords)
        if axis == sp_rank - 1:
            values_left = grid[coords_left]
            values_right = grid[coords_right]
        else:
            values_left = interpolate_nd(is_hi_by_axis, axis + 1)
            values_right = interpolate_nd(is_hi_by_axis_2, axis + 1)
        return spatial_stack([values_left, values_right], axis)

    result = interpolate_nd(np.array([False] * sp_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (not extrap[dim, 0].is_copy_pad, not extrap[dim, 1].is_copy_pad) for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad)
    coordinates += [not extrap[dim, 0].is_copy_pad for dim in grid.shape.spatial.names]
    # --- Find neighbors and interpolate ---
    closest = closest_grid_values(grid, coordinates)
    right_weights = coordinates % 1
    left_weights = 1 - right_weights
    return sum_(closest * weights, axis=grid.shape.spatial.names)


def broadcast_op(operation, tensors):
    non_atomic_dims = set()
    for tensor in tensors:
        if isinstance(tensor, TensorStack) and tensor.keep_separate:
            non_atomic_dims.add(tensor.stack_dim_name)
    if len(non_atomic_dims) == 0:
        return operation(*tensors)
    elif len(non_atomic_dims) == 1:
        dim = next(iter(non_atomic_dims))
        dim_type = None
        size = None
        unstacked = []
        for tensor in tensors:
            if dim in tensor.shape:
                unstacked_tensor = tensor.unstack(dim)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                    dim_type = tensor.shape.get_type(dim)
                else:
                    assert size == len(unstacked_tensor)
                    assert dim_type == tensor.shape.get_type(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(operation(*gathered))
        return TensorStack(result_unstacked, dim, dim_type, keep_separate=True)
    else:
        raise NotImplementedError()


def reshape(value, shape):
    raise NotImplementedError()


def prod(value, axis=None):
    if axis is None and isinstance(value, (tuple, list)) and all(isinstance(v, numbers.Number) for v in value):
        return SCIPY_BACKEND.prod(value)
    raise NotImplementedError()


def divide_no_nan(x, y):
    x = tensor(x)
    return x._op2(y, lambda t1, t2: math.divide_no_nan(t1, t2))


def where(condition, x=None, y=None):
    raise NotImplementedError()


def sum_(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.sum)


def _reduce(value: Tensor or list or tuple, axis, native_function):
    if axis in ((), [], EMPTY_SHAPE):
        return value
    if isinstance(value, (tuple, list)):
        values = [tensor(v) for v in value]
        value = _stack(values, '_reduce', BATCH_DIM)
        if axis is None:
            pass  # continue below
        elif axis == 0:
            axis = '_reduce'
        else:
            raise ValueError('axis must be 0 or None when passing a sequence of tensors')
    else:
        value = tensor(value)
    axes = _axis(axis, value.shape)
    if isinstance(value, NativeTensor):
        result = native_function(value.native(), axis=value.shape.index(axes))
        return NativeTensor(result, value.shape.without(axes))
    elif isinstance(value, TensorStack):
        # --- inner reduce ---
        inner_axes = [ax for ax in axes if ax != value.stack_dim_name]
        red_inners = [_reduce(t, inner_axes, native_function) for t in value.tensors]
        # --- outer reduce ---
        if value.stack_dim_name in axes:
            if any([isinstance(t, SparseLinearOperation) for t in red_inners]):
                return sum_operators(red_inners)  # TODO other functions
            natives = [t.native() for t in red_inners]
            result = native_function(natives, axis=0)
            return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, value.stack_dim_name, value.stack_dim_type, keep_separate=value.keep_separate)
    else:
        raise NotImplementedError()


def _axis(axis, shape: Shape):
    if axis is None:
        return shape.names
    if isinstance(axis, (tuple, list)):
        return axis
    if isinstance(axis, (str, int)):
        return [axis]
    raise ValueError(axis)


def mean(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.mean)


def std(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.std)


def any_(boolean_tensor: Tensor or list or tuple, axis=None):
    return _reduce(boolean_tensor, axis, math.any)


def all_(boolean_tensor: Tensor or list or tuple, axis=None):
    return _reduce(boolean_tensor, axis, math.all)


def max(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.max)


def min(value: Tensor or list or tuple, axis=None):
    return _reduce(value, axis, math.min)


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype)


def ones_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype) + 1


def dot(a, b, axes):
    raise NotImplementedError()


def matmul(A, b):
    raise NotImplementedError()


def einsum(equation, *tensors):
    raise NotImplementedError()


def abs(x: Tensor):
    return x._op1(math.abs)


def sign(x: Tensor):
    return x._op1(math.sign)


def round(x: Tensor):
    return x._op1(math.round)


def ceil(x: Tensor):
    return x._op1(math.ceil)


def floor(x: Tensor):
    return x._op1(math.floor)


def maximum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.maximum)


def minimum(a, b):
    a_, b_ = tensor(a, b)
    return a_._op2(b_, math.minimum)


def clip(x, minimum, maximum):
    def _clip(x, minimum, maximum):
        new_shape, (x_, min_, max_) = broadcastable_native_tensors(*tensor(x, minimum, maximum))
        result_tensor = math.clip(x_, min_, max_)
        return NativeTensor(result_tensor, new_shape)
    return broadcast_op(_clip, tensor(x, minimum, maximum))


def with_custom_gradient(function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
    raise NotImplementedError()


def sqrt(x):
    return tensor(x)._op1(math.sqrt)


def exp(x):
    return tensor(x)._op1(math.exp)


def conv(tensor, kernel, padding='same'):
    raise NotImplementedError()


def shape(tensor):
    return tensor.shape.sizes if isinstance(tensor, Tensor) else math.shape(tensor)


def ndims(tensor):
    return tensor.rank if isinstance(tensor, Tensor) else math.ndims(tensor)


def staticshape(tensor):
    if isinstance(tensor, Tensor):
        return tensor.shape.sizes
    else:
        return math.staticshape(tensor)


def to_float(x, float64=False):
    return tensor(x)._op1(partial(math.to_float, float64=float64))


def to_int(x, int64=False):
    return tensor(x)._op1(partial(math.to_int, int64=int64))


def to_complex(x):
    return tensor(x)._op1(math.to_complex)


def unstack(tensor, axis=0):
    assert isinstance(tensor, Tensor)
    return tensor.unstack(tensor.shape.names[axis])


def boolean_mask(x, mask):
    raise NotImplementedError()


def isfinite(x):
    return tensor(x)._op1(lambda t: math.isfinite(t))


def scatter(points, indices, values, shape, duplicates_handling='undefined'):
    raise NotImplementedError()


def fft(x):
    raise NotImplementedError()


def ifft(k):
    native, assemble = _invertible_standard_form(k)
    result = math.ifft(native)
    return assemble(result)


def imag(complex):
    return complex._op1(math.imag)


def real(complex: Tensor):
    return complex._op1(math.real)


def cast(x: Tensor, dtype):
    return x._op1(lambda t: math.cast(x, dtype))


def sin(x):
    return tensor(x)._op1(math.sin)


def cos(x):
    return tensor(x)._op1(math.cos)


def dtype(x):
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return math.dtype(x)


def tile(value, multiples):
    raise NotImplementedError()


def expand_channel(x, dim_size, dim_name):
    x = tensor(x)
    shape = x.shape.expand(dim_size, dim_name, CHANNEL_DIM)
    return CollapsedTensor(x, shape)


def sparse_tensor(indices, values, shape):
    raise NotImplementedError()


def _invertible_standard_form(tensor: Tensor):
    normal_order = tensor.shape.normal_order()
    native = tensor.native(normal_order.names)
    standard_form = (tensor.shape.batch.volume,) + tensor.shape.spatial.sizes + (tensor.shape.channel.volume,)
    reshaped = math.reshape(native, standard_form)

    def assemble(reshaped):
            un_reshaped = math.reshape(reshaped, math.shape(native))
            return NativeTensor(un_reshaped, normal_order)

    return reshaped, assemble


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.

    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    :param tensors: tensor or tensor-like (constant) each
    :param rel_tolerance: relative tolerance
    :param abs_tolerance: absolute tolerance
    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = math.numpy(native1)
    np2 = math.numpy(native2)
    if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
        np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)


def conjugate_gradient(operator, y, x0, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None, bake='sparse'):
    x0, y = tensor(x0, y)
    batch = combined_shape(y, x0).batch
    x0_native = math.reshape(x0.native(), (x0.shape.batch.volume, x0.shape.non_batch.volume))
    y_native = math.reshape(y.native(), (y.shape.batch.volume, y.shape.non_batch.volume))
    if callable(operator):
        A_ = None
        if bake == 'sparse':
            build_time = time.time()
            x_track = as_sparse_linear_operation(x0)
            try:
                Ax_track = operator(x_track)
                if isinstance(Ax_track, SparseLinearOperation):
                    A_ = Ax_track.dependency_matrix
                    print("CG: matrix build time: %s" % (time.time() - build_time))
                else:
                    warnings.warn("Could not create matrix for conjugate_gradient() because non-linear operations were used.")
            except NotImplementedError as err:
                warnings.warn("Could not create matrix for conjugate_gradient():\n%s" % err)
                raise err
        if A_ is None:
            def A_(native_x):
                x = math.reshape(native_x, x0.shape.non_batch.sizes)
                x = NativeTensor(x, x0.shape.non_batch)
                Ax = operator(x)
                Ax_native = math.reshape(Ax.native(), math.shape(native_x))
                return Ax_native
    else:
        A_ = math.reshape(operator.native(), (y.shape.non_batch.volume, x0.shape.non_batch.volume))

    cg_time = time.time()
    converged, x, iterations = math.conjugate_gradient(A_, y_native, x0_native, relative_tolerance, absolute_tolerance, max_iterations, gradient, callback)
    print("CG: loop time: %s (%s iterations)" % (time.time() - cg_time, iterations))
    converged = math.reshape(converged, batch.sizes)
    x = math.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
    iterations = math.reshape(iterations, batch.sizes)
    return NativeTensor(converged, batch), NativeTensor(x, batch.combined(x0.shape.non_batch)), NativeTensor(iterations, batch)
