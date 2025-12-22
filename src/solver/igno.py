from abc import ABC

from src.components.nf import RealNVP
from src.solver.base import Solver, ProblemInstance
import torch
from src.utils.misc_utils import get_default_device
from typing import Optional, override


class IGNOProblemInstance(ProblemInstance, ABC):

    def __init__(
            self,
            device: torch.device | str = get_default_device(),
            dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype
        )

    @override
    def pre_train_check(self) -> None:
        super().pre_train_check()

        model_keys = {k.lower() for k in self.get_model_dict().keys()}
        if 'nf' not in model_keys:
            raise AssertionError("Model dictionary must contain a key 'nf' (case-insensitive).")

class IGNOSolver(Solver):

    def __init__(
            self,
            problem_instance: IGNOProblemInstance,
    ):
        if not isinstance(problem_instance, IGNOProblemInstance):
            raise TypeError("The provided problem instance must be an instance of IGNOProblemInstance.")

        super().__init__(
            problem_instance=problem_instance
        )

    def learn_mapping(self):
        self.problem_instance.pre_train_check()
        pass

    def invert(self):
        pass


