import torch
from torch.optim.optimizer import Optimizer

class BLOSAM(Optimizer):
    def __init__(self, params, lr=0.001, rho=0.05, adaptive=False, p=2,
        xi_lr_ratio=3, momentum_theta=0.9, weight_decay=0.0, dampening=0):
        if lr <= 0.0 or rho <= 0.0:
            raise ValueError("Invalid learning rate or rho")
        if p not in [2, float('inf')]:
            raise ValueError("Only p=2 and p=inf supported")

        defaults = dict(
            lr = lr, rho = rho,adaptive=adaptive, p=p,
            xi_lr_ratio = xi_lr_ratio,
            momentum_theta = momentum_theta,
            weight_decay=weight_decay,
            dampening = dampening
        )
        super(BLOSAM, self).__init__(params, defaults)


        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['xi'] = torch.zeros_like(p.data)
                # state['v_theta'] = torch.clone(p.grad).detach()
                # state['v_xi'] = torch.zeros_like(p.data)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # grad_norm = self.grad_norm()
        for group in self.param_groups:
            rho = group['rho']
            p_norm = group['p']
            eta_theta = group['lr']
            eta_xi = eta_theta * group['xi_lr_ratio']
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state['old_p']=p.data.clone()
                grad = p.grad.data
                xi = param_state['xi']
                u = xi + grad
                if p_norm == 2:
                    # grad_norm = self._grad_norm()
                    grad_norm = torch.norm(u)
                    projected = u if grad_norm <= rho else rho * u / grad_norm
                else: # p == ∞
                    projected = torch.clamp(u, -rho, rho)
                # print("projected:", projected.abs().max().item())
                xi = xi + eta_xi * (-xi + projected)
                param_state['xi'] = xi
                p.add_(xi)
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            eta_theta = group['lr']
            mu_theta = group['momentum_theta']
            weight_decay = group['weight_decay']
            dampening=group['dampening']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']

                grad = p.grad
                if weight_decay !=0:
                    grad = grad.add(p.data, alpha=weight_decay)
                param_state = self.state[p]
                if mu_theta != 0:
                    if 'v_theta' not in param_state:
                        buf = param_state['v_theta'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['v_theta']
                        buf.mul_(mu_theta).add_(grad, alpha=1-dampening)
                    update = buf
                else:
                    update = grad
                p.add_(update, alpha=-eta_theta)
        
        if zero_grad:
            self.zero_grad()
    
    # def _grad_norm(self):
    #     shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    #     norm = torch.norm(
    #                 torch.cat([
    #                     (self.state[p]['xi']+(torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
    #                     for group in self.param_groups for p in group["params"]
    #                     if p.grad is not None and "xi" in self.state[p]
    #                 ]),
    #                 p=2
    #            )
    #     return norm
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAMPSO requires closure to perform a forward-backward pass"
        closure = torch.enable_grad()(closure)  # Forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

