# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import pickle
import io
import inspect
import copy
import uuid
import types
import dnnlib

#----------------------------------------------------------------------------

_version = 6
_decorators = set() # {decorator_class}
_import_hooks = [] # [function]
_module_to_src_dict = dict() # {module: src}
_src_to_module_dict = dict() # {src: module}

#----------------------------------------------------------------------------

def is_persistent(obj):
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators # pylint: disable=unidiomatic-typecheck

def import_hook(func):
    assert callable(func)
    _import_hooks.append(func)

#----------------------------------------------------------------------------

def persistent_class(orig_class):
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            return dnnlib.EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(type='class', version=_version, module_src=self._orig_module_src, class_name=self._orig_class_name, state=fields[2])
                fields[0] = _reconstruct_persistent_obj # reconstruct func
                fields[1] = (meta,) # reconstruct args
                fields[2] = None # state dict
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    _decorators.add(Decorator)
    return Decorator

#----------------------------------------------------------------------------

def _reconstruct_persistent_obj(meta):
    meta = dnnlib.EasyDict(meta)
    meta.state = dnnlib.EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == 'class'
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, '__setstate__', None)
    if callable(setstate):
        setstate(meta.state) # pylint: disable=not-callable
    else:
        obj.__dict__.update(meta.state)
    return obj

#----------------------------------------------------------------------------

def _module_to_src(module):
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src

def _src_to_module(src):
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__) # pylint: disable=exec-used
    return module

#----------------------------------------------------------------------------

def _check_pickleable(obj):
    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None # Primitive types are pickleable.
        if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor']:
            return None # Tensors are pickleable.
        if is_persistent(obj):
            return None # Persistent objects are pickleable, by virtue of the constructor check.
        return obj
    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)

#----------------------------------------------------------------------------
