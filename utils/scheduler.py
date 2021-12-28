import math
import torch.optim as optim


class WarmUp:
    def __init__(self, base_value, max_warmup_iter):
        # base_value: float, maximum value
        # max_warmup_iter: int, maximum warmup iteration
        self.base_value = base_value
        self.max_warmup_iter = max_warmup_iter


class ExpWarmup(WarmUp):
    def __call__(self, cur_step):
        """exponential warmup proposed in mean teacher
        base_value * exp(-5(1 - t)^2), t = cur_step / max_warmup_iter
        Parameters
        -----
        cur_step: int
            current iteration
        """
        if self.max_warmup_iter <= cur_step:
            return self.base_value
        return self.base_value * math.exp(-5 * (1 - cur_step / self.max_warmup_iter) ** 2)


class NoWarmup(WarmUp):
    # No warmup
    def __call__(self, *args):
        return 1


class LinearWarmup(WarmUp):
    def __call__(self, cur_step):
        """linear warmup
        base_value * (cur_step / max_warmup_iter)
        Parameters
        -----
        cur_step: int
            current iteration
        """
        if self.max_warmup_iter <= cur_step:
            return self.base_value
        return self.base_value * cur_step / self.max_warmup_iter


def cosine_decay(base_lr, max_iteration, cur_step):
    """cosine learning rate decay
    cosine learning rate decay with parameters proposed FixMatch
    base_lr * cos( (7\pi cur_step) / (16 max_warmup_iter) )
    Parameters
    -----
    base_lr: float
        maximum learning rate
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    return base_lr * (math.cos((7 * math.pi * cur_step) / (16 * max_iteration)))


def CosineAnnealingLR(optimizer, max_iteration):
    """
    generate cosine annealing learning rate scheduler as LambdaLR
    """
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda cur_step: math.cos(
        (7 * math.pi * cur_step) / (16 * max_iteration)))


WARMUP_SCHEDULER = {
    'linear': LinearWarmup,
    'exp': ExpWarmup,
    'no_warmup': NoWarmup,
}


if __name__ == '__main__':
    sch = ExpWarmup(1e-1, 0)
    import matplotlib
    matplotlib.use('TkAgg')
    import pylab as plt
    y = [sch(i) for i in range(10000)]
    plt.plot(y)
    plt.show()