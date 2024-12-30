# reference: https://github.com/jianglongye/featurenerf/blob/main/src/model/resnetfc.py

import torch
import torch.autograd.profiler as profiler
from torch import nn
from src.models.utils import combine_interleaved


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type="average",
        use_spade=False,
        use_layer_norm=False,
        alpha=1.0,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade
        self.use_layer_norm = use_layer_norm
        self.alpha = alpha

        self.blocks = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)])
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_blocks)])

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)])
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)])
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(self.alpha))  # i.e 0.1 -> Gradual latent injection

    def forward(self, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None, ret_last_feat=False):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx
            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    # The following implements camera frustum culling, requires torch_scatter
                    #  if combine_index is not None:
                    #      combine_type = (
                    #          "mean"
                    #          if self.combine_type == "average"
                    #          else self.combine_type
                    #      )
                    #      if dim_size is not None:
                    #          assert isinstance(dim_size, int)
                    #      x = torch_scatter.scatter(
                    #          x,
                    #          combine_index,
                    #          dim=0,
                    #          dim_size=dim_size,
                    #          reduce=combine_type,
                    #      )
                    #  else:
                    x = combine_interleaved(x, combine_inner_dims, self.combine_type)

                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + self.alpha * tz
                    else:
                        x = x + self.alpha * tz

                x = self.blocks[blkid](x)
                if self.use_layer_norm:
                    x = self.layer_norms[blkid](x)
            out = self.lin_out(self.activation(x))
            if not ret_last_feat:
                return out
            else:
                return torch.cat([out, x], dim=-1)
    
    @classmethod
    def from_conf(cls, cfg, d_in, d_latent):
        return cls(
            d_in=d_in,  # point dim = 3
            d_out=cfg.d_out_geo + cfg.d_out_sem,  # originally: r, g, b, sigma
            n_blocks=cfg.n_blocks,
            d_latent=d_latent,  # feature dim from encoder
            d_hidden=cfg.d_hidden,
            beta=cfg.beta,
            combine_layer=cfg.combine_layer,
            combine_type=cfg.combine_type,
            use_spade=cfg.use_spade,
            use_layer_norm=cfg.use_layer_norm,
            alpha=cfg.alpha
        )
