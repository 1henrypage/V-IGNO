from src.components.nf import RealNVP
from src.solver.base import Solver, ProblemInstance


class IGNOSolver(Solver):

    def __init__(
            self,
            problem_instance: ProblemInstance,
            beta_dim: int
    ):
        super().__init__(
            problem_instance=problem_instance
        )

        self.real_nvp = RealNVP(
            latent_dim=128

        )

    def learn_mapping(self):
        pass

    def invert(self):
        pass


