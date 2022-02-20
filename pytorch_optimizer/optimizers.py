from pytorch_optimizer.adabelief import AdaBelief
from pytorch_optimizer.adabound import AdaBound
from pytorch_optimizer.adahessian import AdaHessian
from pytorch_optimizer.adamp import AdamP
from pytorch_optimizer.diffgrad import DiffGrad
from pytorch_optimizer.diffrgrad import DiffRGrad
from pytorch_optimizer.lamb import Lamb
from pytorch_optimizer.lars import LARS
from pytorch_optimizer.madgrad import MADGRAD
from pytorch_optimizer.radam import RAdam
from pytorch_optimizer.ralamb import RaLamb
from pytorch_optimizer.ranger import Ranger
from pytorch_optimizer.ranger21 import Ranger21
from pytorch_optimizer.sgdp import SGDP


def load_optimizers(optimizer: str):
    optimizer: str = optimizer.lower()

    if optimizer == 'adamp':
        opt = AdamP
    elif optimizer == 'ranger':
        opt = Ranger
    elif optimizer == 'ranger21':
        opt = Ranger21
    elif optimizer == 'sgdp':
        opt = SGDP
    elif optimizer == 'radam':
        opt = RAdam
    elif optimizer == 'adabelief':
        opt = AdaBelief
    elif optimizer == 'adabound':
        opt = AdaBound
    elif optimizer == 'madgrad':
        opt = MADGRAD
    elif optimizer == 'diffgrad':
        opt = DiffGrad
    elif optimizer == 'diffrgrad':
        opt = DiffRGrad
    elif optimizer == 'adahessian':
        opt = AdaHessian
    elif optimizer == 'lamb':
        opt = Lamb
    elif optimizer == 'ralamb':
        opt = RaLamb
    elif optimizer == 'lars':
        opt = LARS
    else:
        raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')

    return opt
