# 🚨🚨🚨 Experimental parallelism support for Diffusers 🚨🚨🚨
# Experimental changes are subject to change and APIs may break without warning.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch

from ..utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)  # pylint: disable=invalid-name

# TODO(aryan): add support for the following:
# - Unified Attention
# - More dispatcher attention backends
# - CFG/Data Parallel
# - Tensor Parallel

@dataclass
class ContextParallelConfig:


    ring_degree: Optional[int] = None
    ulysses_degree: Optional[int] = None
    convert_to_fp32: bool = True
    # TODO: support alltoall
    rotate_method: Literal["allgather", "alltoall"] = "allgather"

    _rank: int = None
    _world_size: int = None
    _device: torch.device = None
    _mesh: torch.distributed.device_mesh.DeviceMesh = None
    _flattened_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ulysses_mesh: torch.distributed.device_mesh.DeviceMesh = None
    _ring_local_rank: int = None
    _ulysses_local_rank: int = None

    def __post_init__(self):
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1

        if self.ring_degree == 1 and self.ulysses_degree == 1:
            raise ValueError(
                "Either ring_degree or ulysses_degree must be greater than 1 in order to use context parallel inference"
            )
        if self.ring_degree < 1 or self.ulysses_degree < 1:
            raise ValueError("`ring_degree` and `ulysses_degree` must be greater than or equal to 1.")
        if self.rotate_method != "allgather":
            raise NotImplementedError(
                f"Only rotate_method='allgather' is supported for now, but got {self.rotate_method}."
            )

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        return (self.ring_degree, self.ulysses_degree)

    @property
    def mesh_dim_names(self) -> Tuple[str, str]:

        return ("ring", "ulysses")

    def setup(self, rank: int, world_size: int, device: torch.device, mesh: torch.distributed.device_mesh.DeviceMesh):
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._mesh = mesh

        if self.ulysses_degree * self.ring_degree > world_size:
            raise ValueError(
                f"The product of `ring_degree` ({self.ring_degree}) and `ulysses_degree` ({self.ulysses_degree}) must not exceed the world size ({world_size})."
            )

        self._flattened_mesh = self._mesh._flatten()
        self._ring_mesh = self._mesh["ring"]
        self._ulysses_mesh = self._mesh["ulysses"]
        self._ring_local_rank = self._ring_mesh.get_local_rank()
        self._ulysses_local_rank = self._ulysses_mesh.get_local_rank()

@dataclass
class ParallelConfig:


    context_parallel_config: Optional[ContextParallelConfig] = None

    _rank: int = None
    _world_size: int = None
    _device: torch.device = None
    _mesh: torch.distributed.device_mesh.DeviceMesh = None

    def setup(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
    ):
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._mesh = mesh
        if self.context_parallel_config is not None:
            self.context_parallel_config.setup(rank, world_size, device, mesh)

@dataclass(frozen=True)
class ContextParallelInput:


    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False

    def __repr__(self):
        return f"ContextParallelInput(split_dim={self.split_dim}, expected_dims={self.expected_dims}, split_output={self.split_output})"

@dataclass(frozen=True)
class ContextParallelOutput:


    gather_dim: int
    expected_dims: Optional[int] = None

    def __repr__(self):
        return f"ContextParallelOutput(gather_dim={self.gather_dim}, expected_dims={self.expected_dims})"

# A dictionary where keys denote the input to be split across context parallel region, and the
# value denotes the sharding configuration.
# If the key is a string, it denotes the name of the parameter in the forward function.
# If the key is an integer, split_output must be set to True, and it denotes the index of the output
# to be split across context parallel region.
ContextParallelInputType = Dict[
    Union[str, int], Union[ContextParallelInput, List[ContextParallelInput], Tuple[ContextParallelInput, ...]]
]

# A dictionary where keys denote the output to be gathered across context parallel region, and the
# value denotes the gathering configuration.
ContextParallelOutputType = Union[
    ContextParallelOutput, List[ContextParallelOutput], Tuple[ContextParallelOutput, ...]
]

# A dictionary where keys denote the module id, and the value denotes how the inputs/outputs of
# the module should be split/gathered across context parallel region.
ContextParallelModelPlan = Dict[str, Union[ContextParallelInputType, ContextParallelOutputType]]

# Example of a ContextParallelModelPlan (QwenImageTransformer2DModel):
#
# Each model should define a _cp_plan attribute that contains information on how to shard/gather
# tensors at different stages of the forward:
#
# ```python
# _cp_plan = {
#     "": {
#         "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states": ContextParallel...
#         "encoder_hidden_states_mask": ContextPar...
#     },
#     "pos_embed": {
#         0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#         1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#     },
#     "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
# }
# ```
#
# The dictionary is a set of module names mapped t...
# split/gathered according to this at the respective module level. Here, the following happens:
# - "":
#     we specify that we want to split the various...
#     the actual forward logic of the QwenImageTransformer2DModel is run, we will splitthe inputs)
# - "pos_embed":
#     we specify that we want to split the outputs...
#     we can individually specify how they should be split
# - "proj_out":
#     before returning to the user, we gather the...
#     layer forward has run).
#
# ContextParallelInput:
#     specifies how to split the input tensor in t...
#
# ContextParallelOutput:
#     specifies how to gather the input tensor in...
