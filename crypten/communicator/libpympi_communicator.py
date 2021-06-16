import libpympi
import torch
import numpy as np


class LibpympiSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            libpympi.mpi_init()
            print("mpi_init")

        return cls._instance

    def __init__(self):
        pass

    def __del__(self):
        libpympi.mpi_finalize()
        print("mpi_del")


def tensor_allreduce(tensor):
    # a = torch.randint(low=0, high=2**63-1, size=(2, 3), dtype=torch.int64)
    nparray = tensor.cpu().detach().numpy()
    npbytes = nparray.tobytes()
    npshape = nparray.shape

    ret_bytes = libpympi.mpi_allreduce(npbytes)

    ret = np.frombuffer(ret_bytes, dtype=np.int64)
    ret = ret.reshape(npshape)

    return torch.tensor(np.copy(ret))


def mpi_all_reduce(input, batched=False):
    """Reduces the input data across all parties; all get the final result."""
    if batched:
        assert isinstance(
            input, list), "batched reduce input must be a list"
        for x in input:
            result += [tensor_allreduce(x.data)]
    else:
        assert torch.is_tensor(
            input.data
        ), "unbatched input for reduce must be a torch tensor"
        result = tensor_allreduce(input.data)
    return result


_mpi_init = LibpympiSingleton()
