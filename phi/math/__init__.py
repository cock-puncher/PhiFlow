"""
The phi.math package is the main API for tensor / array operations in PhiFlow.
It provides a common inferface for mathematical operations over tensors which currently supports NumPy, TensorFlow and PyTorch.

Provides

* A tensor base class with multiple implementations
* A NumPy-like API for mathematical operations over tensors as well as tensor generation

The provided operations are not implemented directly.
Instead, they delegate the actual computation to either NumPy, TensorFlow or PyTorch, depending on the configuration.
This allows the user to write simulation code once and have it run with various computation backends.

Main classes:

* Tensor
* Shape
"""

from .backend import DYNAMIC_BACKEND, extrapolation, Extrapolation
from .backend._scipy_backend import SCIPY_BACKEND

from phi.struct.struct_backend import StructBroadcastBackend

from ._shape import Shape, define_shape, spatial_shape, infer_shape, EMPTY_SHAPE
from ._tensors import tensor, Tensor, combined_shape, Tensor as Tensor
from ._tensor_math import (
    is_tensor, as_tensor,
    copy,
    transpose,
    zeros, ones, fftfreq, random_normal, meshgrid,  # creation operators (use default backend)
    batch_stack, spatial_stack, channel_stack,
    concat,
    pad, spatial_pad,
    resample,
    reshape,
    prod,
    divide_no_nan,
    where,
    sum, mean, std,
    zeros_like, ones_like,
    dot,
    matmul,
    einsum,
    abs,
    sign,
    round, ceil, floor,
    max, min, maximum, minimum,
    clip,
    with_custom_gradient,
    sqrt,
    exp,
    conv,
    shape, staticshape, ndims,
    to_float, to_int, to_complex,
    unstack,
    boolean_mask,
    isfinite,
    scatter,
    any, all,
    fft, ifft,
    imag, real,
    cast,
    sin, cos,
    dtype,
    tile, expand_channel,
    sparse_tensor,
    close, assert_close,
    conjugate_gradient,
)
from ._nd import (
    shift,
    indices_tensor,
    normalize_to,
    l1_loss, l2_loss, l_n_loss, frequency_loss,
    gradient, laplace,
    fourier_laplace, fourier_poisson, abs_square,
    downsample2x, upsample2x, interpolate_linear,
    spatial_sum, vec_abs, vec_squared
)

# Setup Backend
DYNAMIC_BACKEND.add_backend(StructBroadcastBackend(DYNAMIC_BACKEND))

choose_backend = DYNAMIC_BACKEND.choose_backend