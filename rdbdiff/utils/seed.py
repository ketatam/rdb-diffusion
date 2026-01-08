import random
from typing import Optional

import numpy as np
import torch


def manual_seed(seed: Optional[int]):
    """Seed all python, numpy and pytorch RNGs.

    We derive different per-library seeds from the root seed, to avoid correlated samples in case,
    for example, numpy and CPU torch use the same algorithm.

    Note that all pytorch devices, e.g. CPU, CUDA, etc., are seeded with the same value. This
    ensures that, for example, model weights are initialized consistently in multi-GPU training. If
    you want different random values per device, manage `torch.Generator`s manually.
    """

    root_ss = np.random.SeedSequence(seed)
    std_ss, np_ss, torch_ss = root_ss.spawn(3)

    # Python uses a Mersenne twister with 624 words of state, so we provide enough seed to
    # initialize it fully
    random.seed(std_ss.generate_state(624).tobytes())

    # We seed the global RNG anyway in case some library uses it internally
    np.random.seed(int(np_ss.generate_state(1, np.uint32).item()))

    # Pytorch takes a uint64 seed
    torch.manual_seed(int(torch_ss.generate_state(1, np.uint64).item()))
