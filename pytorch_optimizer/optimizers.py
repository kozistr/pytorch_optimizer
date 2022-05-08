from pytorch_optimizer.adabelief import AdaBelief
from pytorch_optimizer.adabound import AdaBound
from pytorch_optimizer.adamp import AdamP
from pytorch_optimizer.adapnm import AdaPNM
from pytorch_optimizer.diffgrad import DiffGrad
from pytorch_optimizer.diffrgrad import DiffRGrad
from pytorch_optimizer.lamb import Lamb
from pytorch_optimizer.lars import LARS
from pytorch_optimizer.madgrad import MADGRAD
from pytorch_optimizer.nero import Nero
from pytorch_optimizer.pnm import PNM
from pytorch_optimizer.radam import RAdam
from pytorch_optimizer.ralamb import RaLamb
from pytorch_optimizer.ranger import Ranger
from pytorch_optimizer.ranger21 import Ranger21
from pytorch_optimizer.sgdp import SGDP
from pytorch_optimizer.shampoo import Shampoo


def load_optimizers(optimizer: str):  # pylint: disable=R0911
    optimizer: str = optimizer.lower()

    if optimizer == 'adamp':
        return AdamP
    if optimizer == 'ranger':
        return Ranger
    if optimizer == 'ranger21':
        return Ranger21
    if optimizer == 'sgdp':
        return SGDP
    if optimizer == 'radam':
        return RAdam
    if optimizer == 'adabelief':
        return AdaBelief
    if optimizer == 'adabound':
        return AdaBound
    if optimizer == 'madgrad':
        return MADGRAD
    if optimizer == 'diffgrad':
        return DiffGrad
    if optimizer == 'diffrgrad':
        return DiffRGrad
    if optimizer == 'lamb':
        return Lamb
    if optimizer == 'ralamb':
        return RaLamb
    if optimizer == 'lars':
        return LARS
    if optimizer == 'shampoo':
        return Shampoo
    if optimizer == 'pnm':
        return PNM
    if optimizer == 'adapnm':
        return AdaPNM
    if optimizer == 'nero':
        return Nero

    raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')
