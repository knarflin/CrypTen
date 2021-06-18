import libpympi
import torch
import numpy as np
import atexit


class LibpympiSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._mpi_init()
            atexit.register(cls._mpi_finalize)
        return cls._instance

    def __init__(self):
        pass

    def __del__(self):
        pass

    @classmethod
    def _mpi_init(cls):
        libpympi.mpi_init()
        print("mpi_init")  # debug point

    @classmethod
    def _mpi_finalize(cls):
        libpympi.mpi_finalize()
        print("mpi_finalize")  # debug point


def _tensor_all_reduce_sum(tensor):
    # a = torch.randint(low=0, high=2**63-1, size=(2, 3), dtype=torch.int64)
    nparray = tensor.cpu().detach().numpy()
    npbytes = nparray.tobytes()
    npshape = nparray.shape

    ret_bytes = libpympi.mpi_all_reduce_sum(npbytes)

    ret = np.frombuffer(ret_bytes, dtype=np.int64)
    ret = ret.reshape(npshape)

    return torch.tensor(np.copy(ret))


def _tensor_all_reduce_bxor(tensor):
    # a = torch.randint(low=0, high=2**63-1, size=(2, 3), dtype=torch.int64)
    nparray = tensor.cpu().detach().numpy()
    npbytes = nparray.tobytes()
    npshape = nparray.shape

    ret_bytes = libpympi.mpi_all_reduce_bxor(npbytes)

    ret = np.frombuffer(ret_bytes, dtype=np.int64)
    ret = ret.reshape(npshape)

    return torch.tensor(np.copy(ret))


def mpi_all_reduce_sum(input, batched=False):
    """Reduces the input data across all parties; all get the final result."""
    if batched:
        assert isinstance(input, list), "batched reduce input must be a list"
        result = []
        for x in input:
            result += [_tensor_all_reduce_sum(x.data)]
    else:
        assert torch.is_tensor(
            input.data
        ), "unbatched input for reduce must be a torch tensor"
        result = _tensor_all_reduce_sum(input.data)
    return result


def mpi_all_reduce_bxor(input, batched=False):
    """Reduces the input data across all parties; all get the final result."""
    if batched:
        assert isinstance(input, list), "batched reduce input must be a list"
        result = []
        for x in input:
            result += [_tensor_all_reduce_bxor(x.data)]
    else:
        assert torch.is_tensor(
            input.data
        ), "unbatched input for reduce must be a torch tensor"
        result = _tensor_all_reduce_bxor(input.data)
    return result


_mpi_init = LibpympiSingleton()
