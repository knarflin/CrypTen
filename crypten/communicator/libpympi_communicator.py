import libpympi
import torch
import numpy as np
import atexit
import crypten.communicator as comm


class LibpympiSingleton:
    _instance = None
    AR_SIZE_THRESHOLD = 2**16 # Bytes
    AR_ELEMENT_SIZE = 8 # Bytes
    AR_ELEMENT_THRESHOLD =  AR_SIZE_THRESHOLD // AR_ELEMENT_SIZE

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
    nparray = tensor.cpu().detach().numpy()
    npbytes = nparray.tobytes()
    npshape = nparray.shape

    ret_bytes = libpympi.mpi_all_reduce_sum(npbytes)

    ret = np.frombuffer(ret_bytes, dtype=np.int64)
    ret = ret.reshape(npshape)

    return torch.tensor(np.copy(ret))


def _tensor_all_reduce_bxor(tensor):
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
            element_count = np.prod(list(x.shape))
            if element_count <= LibpympiSingleton.AR_ELEMENT_THRESHOLD:
                result += [_tensor_all_reduce_sum(x.data)]
            else:
                result += [comm.get().all_reduce(x.data)]

    else:
        assert torch.is_tensor(
            input.data
        ), "unbatched input for reduce must be a torch tensor"
        element_count = np.prod(list(input.shape))
        if element_count <= LibpympiSingleton.AR_ELEMENT_THRESHOLD:
            result = _tensor_all_reduce_sum(input.data)
        else:
            result = comm.get().all_reduce(input.data)
    return result


def mpi_all_reduce_bxor(input, batched=False):
    """Reduces the input data across all parties; all get the final result."""
    if batched:
        assert isinstance(input, list), "batched reduce input must be a list"
        result = []
        for x in input:
            # result += [_tensor_all_reduce_bxor(x.data)]
            element_count = np.prod(list(x.shape))
            if element_count <= LibpympiSingleton.AR_ELEMENT_THRESHOLD:
                result += [_tensor_all_reduce_sum(x.data)]
            else:
                result += [comm.get().all_reduce(x.data, op=torch.distributed.ReduceOp.BXOR)]
    else:
        assert torch.is_tensor(
            input.data
        ), "unbatched input for reduce must be a torch tensor"
        # result = _tensor_all_reduce_bxor(input.data)
        element_count = np.prod(list(input.shape))
        if element_count <= LibpympiSingleton.AR_ELEMENT_THRESHOLD:
            result = _tensor_all_reduce_bxor(input.data)
        else:
            result = comm.get().all_reduce(input.data, op=torch.distributed.ReduceOp.BXOR)
    return result


_mpi_init = LibpympiSingleton()
