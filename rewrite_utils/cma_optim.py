import cma
import numpy as np
import torch

import types


class CMA:
    """Wrapper for CMA-ES optimizer that mimics more closely follows the step() and backward()
    torch api. Example psuedo usage:

    z = torch.randn(batch_size, zdim)
    params = {'z': z}
    cmaes_optim = CMA(params)
    loss = clip_loss(gan_model(z))
    cmaes_optim.backward(loss)
    cmaes_optim.step()


    """

    _init_idx = 0

    def __init__(
        self,
        params,
        sigma0=0.5,
        popsize=50,
        seed=0,
        AdaptSigma=True,
        CMA_diagonal=False,
        CMA_active=True,
        CMA_elitist=False,
        bounds=None,
        device='cuda',
    ):
        if isinstance(params, types.GeneratorType):
            params = list(params)
        if isinstance(params, list):
            params = {i: v for i, v in enumerate(params)}

        self.params = params
        self.params_full_sizes = {n: p.shape for n, p in params.items()}
        # self.params_concat = np.hstack([p.detach().cpu() for p in params.values()])
        self.params_concat = np.hstack([p.view(p.size(0), -1)[:1].detach().cpu() for p in params.values()])
        self.params_sizes = {n: p.view(p.size(0), -1).shape[-1] for n, p in params.items()}
        self.sigma0 = sigma0
        self.device = device
        self.cma_opts = {
            'popsize': popsize,
            'seed': seed,
            'AdaptSigma': AdaptSigma,
            'CMA_diagonal': CMA_diagonal,
            'CMA_active': CMA_active,
            'CMA_elitist': CMA_elitist,
            'bounds': bounds,
        }
        initial_params = (
            self.params_concat[self._init_idx]
            if self.params_concat.ndim > 1
            else self.params_concat
        )
        self.cmaes_optim = cma.CMAEvolutionStrategy(
            initial_params, self.sigma0, inopts=self.cma_opts
        )

    def step(self):
        cma_results = torch.tensor(self.cmaes_optim.ask(), dtype=torch.float32).to(
            self.device
        )
        param_splits = torch.split_with_sizes(
            cma_results, tuple(self.params_sizes.values()), dim=-1
        )
        for name, val in zip(self.params_sizes, param_splits):
            self.params[name].data = val

        self.params_concat = np.hstack([p.detach().cpu() for p in self.params.values()])

    def backward(self, loss):
        self.cmaes_optim.tell(self.params_concat, loss)
