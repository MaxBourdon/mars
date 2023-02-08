import torch
import numpy as np
import torch.nn as nn
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from functools import partial


class MARS(nn.Module):
    def __init__(self, tensorized_model, 
                 pi=1e-2, alpha=-1.0,
                 temperature=0.1, sigma_inv=0.1, 
                 gamma=-0.1, zeta=1.1,
                 eval_sample=False, ste=False,
                 eval_logits_threshold=0.0):
        """
        MARS wrapping module.
        Given a tensorized model, it applies MARS to it and automatically selects tensor ranks.

        Parameters are:
        tensorized_model : TensorizedModel
            The tensorized model to apply MARS to.
        pi : float
            Prior parameter in Bernoulli masks prior.
        alpha : float
            Phi logits initialization mean.
        temperature : float
            Relaxed Bernoulli temperature.
        sigma_inv : float
            Inverted scale of the cores normal prior.
        gamma : float
            Hard Concrete interval lower bound.
        zeta : float
            Hard Concrete interval upper bound.
        eval_sample : bool
            Whether to sample during evaluation or take the MAP estimate.
        ste : bool
            Whether to use the Straight-Through Estimator (STE)
        eval_logits_threshold : float
            The logits rounding threshold.
        """
        super().__init__()

        self.tensorized_model = tensorized_model

        self.log_prior_prob = np.log(pi)
        self.log_prior_prob_c = np.log(1.0 - pi)

        self.temperature = temperature  # relaxed Bernoulli temperature
        self.l2_weight = 0.5 * sigma_inv ** 2  # cores L2-regularization weight

        # Hard Concrete constants:
        self.gamma = gamma
        self.zeta = zeta
        self.zmg = self.zeta - self.gamma

        self.eval_sample = eval_sample  # Whether to sample during evaluation or take MAP
        self.ste = ste  # Whether to use STE
        self.eval_logits_threshold = eval_logits_threshold  # mask_MAP = [l > ELT for l in logits]

        self.F = partial(logits_to_probs, is_binary=True)  # logits -> probs function (CDF)
        self.F_inv = partial(probs_to_logits, is_binary=True)  # probs -> logits function (iCDF)
        self.warmup = False  # when warmup is True, model doesn't apply masking

        self.phi_logits_list = []        
        for R in self.tensorized_model.ranks:
            logits = nn.Parameter(torch.Tensor(R))
            logits.data.normal_(alpha, 1e-2)
            self.phi_logits_list.append(logits)
        self.phi_logits_list = nn.ParameterList(self.phi_logits_list)

    def get_mask(self, logits):
        "Get masks given phi logits."
        if self.eval_sample or self.training:
            u = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))  # Uniform noise
            logits = logits + self.F_inv(u)  # new logits = old logits + noise logits

            if self.training:
                # Sample soft masks according to Hard Concrete Bernoulli
                s = self.F(logits / self.temperature)
                s = s * self.zmg + self.gamma
                s = torch.clamp(s, min=0.0, max=1.0)

                if self.ste:
                    # STE: differentiable hard sampling
                    s_ste = torch.round(s)  # hard sample
                    s = (s_ste - s).detach() + s  # set the gradients w.r.t. `s` as the gradients w.r.t. `s_ste`

                return s  # return soft masks

        return logits > self.eval_logits_threshold  # return hard masks

    def forward(self, x):
        if self.warmup:
            # no masking during warmup
            return self.tensorized_model(x)

        masks = [self.get_mask(logits) for logits in self.phi_logits_list]
        return self.tensorized_model(x, masks)
        
    def compute_reg(self):
        "Compute the MARS regularizer term: log p(m) + log p(G)."
        reg = 0.0
        probs_list = [self.F(logits) for logits in self.phi_logits_list]
        
        for probs in probs_list:
            # log p(m) term
            probs_c = 1 - probs
            reg += torch.sum(probs * self.log_prior_prob + probs_c * self.log_prior_prob_c)  
            
        if self.l2_weight > 0:
            # log p(G) term
            reg -= self.l2_weight * sum(torch.sum(core ** 2) for core in self.tensorized_model.cores)

        return reg

def compute_cum_reg(model):
    "Compute the cumulative MARS regularizer term of the model."
    reg = 0.0
    
    for layer in model.modules():
        if isinstance(layer, MARS):
            reg += layer.compute_reg()
            
    return reg

class MARSLoss(nn.Module):
    def __init__(self, model, train_size, criterion=None, reg_term_coef=1.0):
        """
        MARS regularized loss.

        Parameters are:
        model : MARS
            The tensorized MARS model.
        train_size : int
            The number of training samples.
        criterion : nn.Module
            The criterion calculating log-likelihood.
        reg_term_coef : float
            Regularizer term multiplicative coefficient.
        """
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.reg_term_coef = reg_term_coef / train_size  # regularization term coefficient
        
    def forward(self, output, target):
        neg_data_term = self.criterion(output, target)
        reg_term = compute_cum_reg(self.model)
        return neg_data_term - self.reg_term_coef * reg_term

def get_MARS_attr(model, attr_name):
    "Get the value of the attribute from (the first) MARS layer in the model."
    for layer in model.modules():
        if isinstance(layer, MARS):
            return getattr(layer, attr_name)

def set_MARS_attr(model, attr_name, attr_value):
    "Set the attribute to the value for all MARS layers in the model."
    for layer in model.modules():
        if isinstance(layer, MARS):
            setattr(layer, attr_name, attr_value)
