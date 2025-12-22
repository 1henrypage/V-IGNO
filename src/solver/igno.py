from abc import ABC

from src.components.nf import RealNVP
from src.solver.base import Solver, ProblemInstance
import torch
from src.utils.misc_utils import get_default_device
from typing import Optional, override


# class IGNOProblemInstance(ProblemInstance, ABC):
#
#     def __init__(
#             self,
#             device: torch.device | str = get_default_device(),
#             dtype: Optional[torch.dtype] = torch.float32,
#     ) -> None:
#         super().__init__(
#             device=device,
#             dtype=dtype
#         )

class IGNOSolver(Solver):

    def __init__(
            self,
            problem_instance: ProblemInstance,
    ):

        super().__init__(
            problem_instance=problem_instance
        )

    def learn_mapping(self):
        pass

    def invert(self):
        pass


