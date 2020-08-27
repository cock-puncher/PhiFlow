from __future__ import annotations

import warnings

from phi import math


BATCH_DIM = 0
SPATIAL_DIM = 1
CHANNEL_DIM = 2


class Shape:

    def __init__(self, sizes: tuple or list, names: tuple or list, types: tuple or list):
        """

        :param sizes: list of dimension sizes
        :param names: list of dimension names, either strings (spatial, batch) or integers (channel)
        :param types: list of types, all values must be one of (CHANNEL_DIM, SPATIAL_DIM, BATCH_DIM)
        """
        assert len(sizes) == len(names) == len(types), "sizes=%s, names=%s, types=%s" % (sizes, names, types)
        self.sizes = tuple(sizes)
        self.names = tuple(names)
        assert all(isinstance(n, str) for n in names), names
        self.types = tuple(types)

    @property
    def named_sizes(self):
        return zip(self.names, self.sizes)

    @property
    def dimensions(self):
        return zip(self.sizes, self.names, self.types)

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        return item in self.names

    def index(self, name):
        if name is None:
            return None
        if isinstance(name, (list, tuple)):
            return tuple(self.index(n) for n in name)
        if isinstance(name, Shape):
            return tuple(self.index(n) for n in name.names)
        for idx, dim_name in enumerate(self.names):
            if dim_name == name:
                return idx
        raise ValueError("Shape %s does not contain dimension with name '%s'" % (self, name))

    def get_size(self, name):
        if isinstance(name, str):
            return self.sizes[self.names.index(name)]
        elif isinstance(name, (tuple, list)):
            return tuple(self.get_size(n) for n in name)
        else:
            raise ValueError(name)

    def __getattr__(self, name):
        if name == 'names':
            raise AssertionError("Attribute missing: %s" % name)
        if name in self.names:
            return self.get_size(name)
        raise AttributeError("Shape has no attribute '%s'" % (name,))

    def get_type(self, name):
        return self.types[self.names.index(name)]

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self.sizes[selection]
        return Shape([self.sizes[i] for i in selection], [self.names[i] for i in selection], [self.types[i] for i in selection])

    def filtered(self, boolean_mask) -> Shape:
        indices = [i for i in range(self.rank) if boolean_mask[i]]
        return self[indices]

    @property
    def channel(self):
        return self.filtered([t == CHANNEL_DIM for t in self.types])

    @property
    def spatial(self) -> Shape:
        return self.filtered([t == SPATIAL_DIM for t in self.types])

    @property
    def batch(self):
        return self.filtered([t == BATCH_DIM for t in self.types])

    @property
    def non_channel(self):
        return self.filtered([t != CHANNEL_DIM for t in self.types])

    @property
    def non_spatial(self):
        return self.filtered([t != SPATIAL_DIM for t in self.types])

    @property
    def non_batch(self):
        return self.filtered([t != BATCH_DIM for t in self.types])

    @property
    def singleton(self):
        return self.filtered([size == 1 for size in self.sizes])

    @property
    def non_singleton(self):
        return self.filtered([size != 1 for size in self.sizes])

    def mask(self, names):
        if isinstance(names, str):
            names = [names]
        mask = [1 if name in names else 0 for name in self.names]
        return tuple(mask)

    def select(self, *names):
        indices = [self.index(name) for name in names]
        return self[indices]

    def __repr__(self):
        strings = ['%s=%s' % (name, size) for size, name, _ in self.dimensions]
        return '(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        return self.names == other.names and self.types == other.types and self.sizes == other.sizes

    def __ne__(self, other):
        return not self == other

    def normal_order(self):
        sizes = self.batch.sizes + self.spatial.sizes + self.channel.sizes
        names = self.batch.names + self.spatial.names + self.channel.names
        types = self.batch.types + self.spatial.types + self.channel.types
        return Shape(sizes, names, types)

    def combined(self, other, allow_inconsistencies=False):
        """
        Returns a Shape object that both `self` and `other` can be broadcast to.
        If `self` and `other` are incompatible, raises a ValueError.
        :param other: Shape
        :return:
        :raise: ValueError if shapes don't match
        """
        assert isinstance(other, Shape)
        sizes = list(self.batch.sizes)
        names = list(self.batch.names)
        types = list(self.batch.types)

        def _check(size, name):
            self_size = self.get_size(name)
            if size != self_size:
                if not allow_inconsistencies:
                    raise IncompatibleShapes(self, other)
                else:
                    sizes[names.index(name)] = None

        for size, name, type in other.batch.dimensions:
            if name not in names:
                names.insert(0, name)
                sizes.insert(0, size)
                types.insert(0, type)
            else:
                _check(size, name)
        # --- spatial ---
        # spatial dimensions must match exactly or one shape has none
        if self.spatial.rank == 0:
            sizes.extend(other.spatial.sizes)
            names.extend(other.spatial.names)
            types.extend(other.spatial.types)
        elif other.spatial.rank == 0:
            sizes.extend(self.spatial.sizes)
            names.extend(self.spatial.names)
            types.extend(self.spatial.types)
        else:
            sizes.extend(self.spatial.sizes)
            names.extend(self.spatial.names)
            types.extend(self.spatial.types)
            if set(self.spatial.names) != set(other.spatial.names):
                raise IncompatibleShapes(self, other)
            for size, name, type in other.spatial.dimensions:
                _check(size, name)
        # --- channel ---
        # channel dimensions must match exactly or one shape has none
        if self.channel.rank == 0:
            sizes.extend(other.channel.sizes)
            names.extend(other.channel.names)
            types.extend(other.channel.types)
        elif other.channel.rank == 0:
            sizes.extend(self.channel.sizes)
            names.extend(self.channel.names)
            types.extend(self.channel.types)
        else:
            sizes.extend(self.channel.sizes)
            names.extend(self.channel.names)
            types.extend(self.channel.types)
            for size, name, type in other.channel.dimensions:
                if name not in names:
                    names.append(name)
                    sizes.append(size)
                    types.append(type)
                else:
                    _check(size, name)
        return Shape(sizes, names, types)

    def __and__(self, other):
        return self.combined(other)

    def expand_batch(self, size, name, pos=None):
        return self.expand(size, name, BATCH_DIM, pos)

    def expand_spatial(self, size, name, pos=None):
        return self.expand(size, name, SPATIAL_DIM, pos)

    def expand_channel(self, size, name, pos=None):
        return self.expand(size, name, CHANNEL_DIM, pos)

    def expand(self, size, name, dim_type, pos=None):
        """
        Add a dimension to the shape.

        The resulting shape has linear indices.
        """
        if pos is None:
            same_type_dims = self.filtered([t == dim_type for t in self.types])
            if len(same_type_dims) > 0:
                pos = self.index(same_type_dims.names[0])
            else:
                pos = {BATCH_DIM: 0, SPATIAL_DIM: self.batch.rank+1, CHANNEL_DIM: self.rank + 1}[dim_type]
        elif pos < 0:
            pos += self.rank + 1
        sizes = list(self.sizes)
        names = list(self.names)
        types = list(self.types)
        sizes.insert(pos, size)
        names.insert(pos, name)
        types.insert(pos, dim_type)
        return Shape(sizes, names, types)

    def without(self, other):
        if isinstance(other, str):
            return self[[i for i in range(self.rank) if self.names[i] != other]]
        if isinstance(other, (tuple, list)):
            return self[[i for i in range(self.rank) if self.names[i] not in other]]
        elif isinstance(other, Shape):
            return self[[i for i in range(self.rank) if self.names[i] not in other.names]]
        elif other is None:  # subtract all
            return EMPTY_SHAPE
        else:
            raise ValueError(other)

    reduce = without

    @property
    def rank(self):
        return len(self.sizes)

    @property
    def well_defined(self):
        return None not in self.sizes

    def with_sizes(self, sizes):
        return Shape(sizes, self.names, self.types)

    def with_size(self, name, size):
        new_sizes = list(self.sizes)
        new_sizes[self.index(name)] = size
        return self.with_sizes(new_sizes)

    def perm(self, names):
        assert set(names) == set(self.names), 'names must match existing dimensions %s but got %s' % (self.names, names)
        perm = [self.names.index(name) for name in names]
        return perm

    @property
    def volume(self):
        """
        Returns the total number of values contained in a tensor of this shape.
        This is the product of all dimension sizes.
        """
        if None in self.sizes:
            return None
        if self.rank == 0:
            return 1
        return math.prod(self.sizes)

    @property
    def is_empty(self):
        return len(self.sizes) == 0

    def order(self, sequence, default=None):
        """
        If sequence is a dict with dimension names as keys, orders its values according to this shape.

        Otherwise, the sequence is returned unchanged.

        :param sequence: sequence or dict to be ordered
        :type sequence: dict or list or tuple
        :param default: default value used for dimensions not contained in sequence
        :return: ordered sequence of values
        """
        if isinstance(sequence, dict):
            result = [sequence.get(name, default) for name in self.names]
            return result
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence
        else:  # just a constant
            return sequence

    def sequence_get(self, sequence, name):
        if isinstance(sequence, dict):
            return sequence[name]
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence[self.names.index(name)]
        if math.is_tensor(sequence):
            assert math.staticshape(sequence) == (self.rank,)
            return sequence[self.names.index(name)]
        else:  # just a constant
            return sequence

    def after_gather(self, selection_dict):
        result = self
        for name, selection in selection_dict.items():
            if isinstance(selection, int):
                result = result.without(name)
            elif isinstance(selection, slice):
                assert selection.step is None
                start = selection.start or 0
                stop = selection.stop or self.get_size(name)
                if stop < 0:
                    stop += self.get_size(name)
                result = result.with_size(name, stop - start)
            else:
                raise NotImplementedError()
        return result


EMPTY_SHAPE = Shape((), (), ())


class IncompatibleShapes(ValueError):
    def __init__(self, shape1, shape2):
        ValueError.__init__(self, shape1, shape2)


def names(obj, count: int) -> tuple:
    if isinstance(obj, str):
        parts = obj.split(',')
        result = []
        for part in parts:
            part = part.strip()
            if part == '...':
                result.extend([None] * (count - len(parts) - 1))
            elif part == ':':
                result.append(None)
            else:
                result.append(part)
        assert len(result) == count
        return tuple(result)
    elif isinstance(obj, Shape):
        assert len(obj) == count
        return obj.names
    elif isinstance(obj, (tuple, list)):
        assert len(obj) == count
        return tuple(obj)
    raise ValueError(obj)


get_names = names


def define_shape(channels=(), names=None, batch=None, infer_types_if_not_given=False, **spatial):
    """
    Creates a shape from the specified channel, spatial and batch dimensions.

    :param channels: int or (int,)
    :param batch: int or {name: size}
    :param infer_types_if_not_given: if True, detects legacy-style shapes, infers the corresponding types and removes singleton dimensions
    :param spatial:
    :return:
    """
    if isinstance(channels, Shape):
        assert batch is None
        assert len(spatial) == 0
        return channels
    if infer_types_if_not_given and batch is None and len(spatial) == 0 and isinstance(channels, (tuple, list)) and len(channels) >= 3:
        shape = infer_shape(channels)
        return shape.without(shape.non_spatial.singleton)
    sizes = []
    names_ = []
    types = []
    # --- Batch dimensions ---
    if isinstance(batch, int):
        sizes.append(batch)
        names_.append('batch')
        types.append(BATCH_DIM)
    elif isinstance(batch, dict):
        for name, size in batch.items():
            sizes.append(size)
            names_.append(name)
            types.append(BATCH_DIM)
    elif batch is None:
        pass
    else:
        raise ValueError(batch)
    # --- Spatial dimensions ---
    for name, size in spatial.items():
        sizes.append(size)
        names_.append(name)
        types.append(SPATIAL_DIM)
    # --- Channel dimensions ---
    if isinstance(channels, int):
        sizes.append(channels)
        names_.append('vector' if channels == len(spatial) else 'channel')
        types.append(CHANNEL_DIM)
    else:
        for i, channel in enumerate(channels):
            sizes.append(channel)
            names_.append('channel%d' % i)
            types.append(CHANNEL_DIM)
    if names is not None:
        names = get_names(names, len(sizes))
        names_ = [setn or detn for setn, detn in zip(names, names_)]
    return Shape(sizes, names_, types)


def infer_shape(shape, dim_names=None, batch_dims=None, spatial_dims=None, channel_dims=None):
    if isinstance(shape, Shape):
        return shape
    shape = tuple(shape)
    if len(shape) == 0:
        return EMPTY_SHAPE
    # --- Infer dim types ---
    dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    if dims is None:  # could not infer
        channel_dims = 1
        dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
        if dims is None:
            batch_dims = 1
            dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    assert dims is not None, "Could not infer shape from '%s' given constraints batch_dims=%s, spatial_dims=%s, channel_dims=%s" % (shape, batch_dims, spatial_dims, channel_dims)
    batch_dims, spatial_dims, channel_dims = dims
    # --- Construct shape ---
    from phi import geom
    types = [BATCH_DIM] * batch_dims + [SPATIAL_DIM] * spatial_dims + [CHANNEL_DIM] * channel_dims
    if dim_names is not None:
        dim_names = names(dim_names, len(shape))
    if dim_names is None or None in dim_names:
        set_dim_names = dim_names
        dim_names = []
        # --- batch names ---
        if batch_dims == 1:
            dim_names.append('batch')
        else:
            for i in range(batch_dims):
                dim_names.append('batch %d' % (i,))
        # --- spatial names ---
        for i in range(spatial_dims):
            dim_names.append(geom.GLOBAL_AXIS_ORDER.axis_name(i, spatial_dims))
        # --- channel names ---
        if channel_dims == 0:
            pass
        elif channel_dims == 1 and shape[-1] == spatial_dims:
            dim_names.append('vector')
        elif channel_dims == 1:
            dim_names.append('channel')
        else:
            for i in range(channel_dims):
                dim_names.append('channel%d' % i)
        if set_dim_names is not None:
            for i, set_name in enumerate(set_dim_names):
                if set_name is not None:
                    dim_names[i] = set_name
    return Shape(sizes=shape, names=dim_names, types=types)


def _infer_dim_group_counts(rank, constraints: list):
    known_sum = sum([dim or 0 for dim in constraints])
    unknown_count = sum([1 if dim is None else 0 for dim in constraints])
    if known_sum == rank:
        return [dim or 0 for dim in constraints]
    if unknown_count == 1:
        return [rank - known_sum if dim is None else dim for dim in constraints]
    return None


def spatial_shape(sizes):
    """
    If `sizes` is a `Shape`, returns the spatial part of it.

    Otherwise, creates a Shape with the given sizes as spatial dimensions.
    The sizes are assumed to be ordered according to the GLOBAL_AXIS_ORDER and the dimensions are named accordingly.

    :param sizes: list of integers or Shape
    :return: Shape containing only spatial dimensions
    """
    if isinstance(sizes, Shape):
        return sizes.spatial
    else:
        return infer_shape(sizes, batch_dims=0, channel_dims=0)


def channel_shape(sizes):
    if isinstance(sizes, Shape):
        return sizes.channel
    else:
        return infer_shape(sizes, batch_dims=0, spatial_dims=0)


def check_singleton(shape):
    for i, (size, dim_type) in enumerate(zip(shape.sizes, shape.types)):
        if isinstance(size, int) and size == 1 and dim_type != SPATIAL_DIM and check_singleton:
            warnings.warn("Dimension '%s' at index %d of shape %s has size 1. Is this intentional? Singleton dimensions are not supported." % (shape.names[i], i, shape.sizes))
