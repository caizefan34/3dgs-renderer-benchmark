"""Single-birth shared/relative coordinates. No repeated-topology support."""

import torch
import torch.nn.functional as F


def encode_children(raw, count):
    plus, minus = raw[-2*count:-count], raw[-count:]
    return torch.cat(((plus+minus)*0.5, (plus-minus)*0.5), dim=0)


def decode_raw(raw, count):
    shared, relative = raw[-2*count:-count], raw[-count:]
    return torch.cat((raw[:-2*count], shared+relative, shared-relative), dim=0)


class SiblingView:
    """Decode raw variables before the reference model's exact activations.

    Intentionally copies full tensors. Timings must include this prototype cost.
    The underlying model owns encoded leaf parameters and their optimizer.
    """

    def __init__(self, model, count):
        self.model = model
        self.count = count

    @property
    def active_sh_degree(self):
        return self.model.active_sh_degree

    @property
    def get_xyz(self):
        return decode_raw(self.model._xyz, self.count)

    @property
    def get_scaling(self):
        return decode_raw(self.model._scaling, self.count).exp()

    @property
    def get_rotation(self):
        return F.normalize(decode_raw(self.model._rotation, self.count), dim=-1)

    @property
    def get_opacity(self):
        return decode_raw(self.model._opacity, self.count).sigmoid().squeeze(-1)

    @property
    def get_features(self):
        return decode_raw(self.model._shs, self.count)[:, :(self.active_sh_degree+1)**2, :].contiguous()
