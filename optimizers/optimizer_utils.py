import torch.optim as optim
from collections.abc import Iterable
import torch
from torch.optim.optimizer import Optimizer


def set_optimizer(args, model):
    if type(model) == list:
        model_parameters = []
        for i in range(len(model)):
            model_parameters += list(model[i].parameters())
    else:
        model_parameters = model.parameters()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model_parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model_parameters,
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'LARS':
        optimizer = LARS(
            params=model_parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    return optimizer

class LARS(Optimizer):
    """ LARS for PyTorch

    Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1.0).
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coeff (float): trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)
        eps (float): eps for division denominator (default: 1e-8)
        trust_clip (bool): enable LARC trust ratio clipping (default: False)
        always_adapt (bool): always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        trust_coeff: float = 0.001,
        eps: float = 1e-8,
        trust_clip: bool = False,
        always_adapt: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        # because torch.where doesn't handle scalars correctly
        one_tensor = torch.tensor(1.0, device=device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coeff = group['trust_coeff']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # apply LARS LR adaptation, LARC clipping, weight decay
                # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / \
                        (g_norm + w_norm * weight_decay + eps)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.minimum(
                            trust_ratio / group['lr'], one_tensor)
                    grad.add_(p, alpha=weight_decay)
                    grad.mul_(trust_ratio)

                # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss