import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
from torch.distributions.normal import Normal

class LinearReLU(nn.Module):
    
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.norm = nn.LayerNorm(dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.residual = dim_in == dim_out
    
    def forward(self, x):
        x_ = self.proj(x)
        x_ = F.relu(x_)
        x_ = self.norm(x_)
        if self.residual:
            x = x + x_
        else:
            x = x_
        return x


class ConvReLU(nn.Module):
    
    def __init__(self, dim_in, dim_out, kernel_size=1, padding=0, bias=False):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, kernel_size, bias=bias, padding=padding)
        self.norm = nn.InstanceNorm1d(dim_out, affine=True)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.residual = dim_in == dim_out
    
    def forward(self, x):
        x_ = self.proj(x)
        x_ = F.relu(x_)
        x_ = self.norm(x_)
        if self.residual:
            return x + x_
        else:
            return x_


class ConvBlock(nn.Module):
    
    def __init__(self, dim_in, dim_mid, dim_out):
        super().__init__()
        self.in_channels = dim_in
        self.channels = dim_mid
        self.out_channels = dim_out
        
        self.conv1 = ConvReLU(dim_in, dim_mid, 3, 1)
        self.conv2 = ConvReLU(dim_mid, dim_mid)
        self.norm = nn.InstanceNorm1d(dim_mid, affine=True)
        self.conv_out = nn.Conv1d(dim_mid, dim_out, 1, bias=True)
        
        self.residual = dim_in == dim_out
    
    def __repr__(self):
        return f'ConvBlock({self.in_channels}, {self.channels}, {self.out_channels})'
    
    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = self.norm(x_)
        x_ = self.conv_out(x_)
        if self.residual:
            return x + x_
        else:
            return x_


class StridedConvolutionalEncoder(nn.Module):
    
    def __init__(self, channels, kernel_size, num_layers):
        super(StridedConvolutionalEncoder, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        def get_conv_layer():
            return nn.Sequential(
                ConvBlock(channels, channels, channels),
                nn.Conv1d(channels, channels, kernel_size, stride=2, bias=False, padding=1),
            )
        
        self.conv_layers = nn.Sequential(*[get_conv_layer() for _ in range(num_layers)])
    
    def forward(self, x, cond=None):
        x = self.conv_layers(x)
        return x


class TransposedConvolutionalDecoder(nn.Module):
    
    def __init__(self, channels, kernel_size, num_layers):
        super(TransposedConvolutionalDecoder, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        def get_conv_layer():
            return nn.Sequential(
                nn.ConvTranspose1d(channels, channels, kernel_size, stride=2, padding=1, bias=False),
                ConvBlock(channels, channels, channels),
            )
        
        self.conv_layers = nn.Sequential(*[get_conv_layer() for _ in range(num_layers)])
    
    def forward(self, x, cond=None):
        x = self.conv_layers(x)
        return x


def _trunc_normal_(tensor, mean, std, a, b):
    
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    l = norm_cdf((a-mean) / std)
    u = norm_cdf((b-mean) / std)
    
    tensor.uniform_(2*l - 1, 2*u - 1)
    
    tensor.erfinv_()
    
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class PositionalEncoding(nn.Module):
    
    def __init__(self, channels, num_seq):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.num_seq = num_seq
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_seq, channels))
        trunc_normal_(self.pos_encoding, std=0.02)
    
    def forward(self, x):
        return x + self.pos_encoding


class TransformerBottleneck(nn.Module):
    
    def __init__(self, channels, num_layers, num_seq):
        super(TransformerBottleneck, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        
        def get_transformer_layer():
            return (
                PositionalEncoding(channels, num_seq),
                nn.TransformerEncoderLayer(channels, 1, channels, 0.0, activation="relu", batch_first=True),
            )
        
        self.transformer_layers = nn.Sequential(*[l for _ in range(num_layers) for l in get_transformer_layer()])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_layers(x)
        return x


class BaselineModel(nn.Module):
    
    def __init__(self, in_channels, channels, latent_dim, num_layers, kernel_size, recon_type="mse", content_cosine=False, time_resolution=256, losses=['recon']):
        super(BaselineModel, self).__init__()
        self.time_resolution = time_resolution
        self.latent_seqs = time_resolution // (2**num_layers)
        self.in_channels = in_channels
        self.channels = channels
        self.latent_dim = latent_dim
        
        self.encoder_in = nn.Sequential(
            ConvReLU(in_channels, channels),
            ConvBlock(channels, channels, channels),
        )
        
        self.encoder = StridedConvolutionalEncoder(channels, kernel_size, num_layers)
        self.encoder_bottleneck = TransformerBottleneck(channels, num_layers, self.latent_seqs)
        
        self.encoder_out_net = ConvBlock(channels, channels, latent_dim)
        
        self.decoder_in_net = ConvBlock(2 * latent_dim, channels, channels)
        
        self.decoder_bottleneck = TransformerBottleneck(channels, num_layers, self.latent_seqs)
        self.decoder = TransposedConvolutionalDecoder(channels, kernel_size, num_layers)
        
        self.decoder_out = ConvBlock(channels, channels, in_channels)
        
        self.subj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.task_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.effective_latent_dim = self.latent_seqs * self.latent_dim
        
        self.recon_type = recon_type
        self.content_cosine = content_cosine
        
        self.used_losses = losses
        self.batch_size = 256
        
        self.loader = None
        
        self.loss_weights = defaultdict(lambda: 1.0)
        if any(['sub_cross' in l for l in self.used_losses]):
            self.sub_clf_layer = nn.Linear(self.latent_seqs * self.latent_dim, 40)
            self.sub_clf_layer_init = True
        else:
            self.sub_clf_layer = nn.Identity()
            self.sub_clf_layer_init = False
        if any(['task_cross' in l for l in self.used_losses]):
            self.task_clf_layer = nn.Linear(self.latent_seqs * self.latent_dim, 14)
            self.task_clf_layer_init = True
        else:
            self.task_clf_layer = nn.Identity()
            self.task_clf_layer_init = False

    
    def set_losses(self, losses=[], batch_size=256, loader=None, loss_weights=None):
        self.used_losses = [l.lower() for l in losses]
        self.loader = loader
        self.batch_size = batch_size
        if loss_weights is not None:
            self.loss_weights = loss_weights
        if not self.sub_clf_layer_init and any(['sub_cross' in l for l in self.used_losses]):
            self.sub_clf_layer = nn.Linear(self.latent_seqs * self.latent_dim, 40)
            self.sub_clf_layer_init = True
        if not self.task_clf_layer_init and any(['task_cross' in l for l in self.used_losses]):
            self.task_clf_layer = nn.Linear(self.latent_seqs * self.latent_dim, 14)
            self.task_clf_layer_init = True
    
    def pre_encode(self, x):
        raise NotImplementedError
    
    def post_decode(self, x):
        raise NotImplementedError
    
    def encode(self, x):
        x = self.pre_encode(x)
        x = self.encoder_out_net(x)
        x = x.view(-1, self.latent_seqs * self.latent_dim)
        return x
    
    def decode(self, x):
        x = x.view(-1, self.latent_dim, self.latent_seqs)
        x = self.decoder_in_net(x)
        x = self.post_decode(x)
        return x
    
    def reconstruction_loss(self, x, x_hat):
        if x.ndim > 3:
            x = x.view(-1, *x.shape[-2:])
            x_hat = x_hat.view(-1, *x_hat.shape[-2:])
        if self.recon_type.lower().startswith('c'):
            return self.cosine_distance_loss(x, x_hat)
        elif self.recon_type.lower().startswith('m'):
            return F.mse_loss(x, x_hat, reduction="mean")
        elif self.recon_type.lower().startswith('s'):
            std = x.std(dim=(0, 2), keepdim=True).clamp(1e-8)
            return F.mse_loss(x/std, x_hat/std, reduction="mean")
        elif self.recon_type.lower().startswith('n'):
            std = x.std(dim=(0, 2), keepdim=True).clamp(1e-8)
            dist = Normal(x, std)
            return -dist.log_prob(x_hat).mean()
        return F.mse_loss(x, x_hat, reduction="mean")
    
    def cosine_distance_loss(self, x, x_hat):
        return (1 - F.cosine_similarity(x, x_hat, dim=-1)).mean()
    
    def contrastive_process(self, l1, l2, which='subject', norm=True):
        if which.lower().startswith('s'):
            logit_scale = self.subj_logit_scale.exp()
        elif which.lower().startswith('t'):
            logit_scale = self.task_logit_scale.exp()
        else:
            raise ValueError(f"Unknown contrastive loss type: {which}")
        
        if norm:
            l1 = F.normalize(l1, dim=-1, eps=1e-8)
            l2 = F.normalize(l2, dim=-1, eps=1e-8)
        return l1, l2, logit_scale
    
    def contrastive_loss(self, l1, l2, which='subject', norm=True):
        l1, l2, logit_scale = self.contrastive_process(l1, l2, which, norm)
        logits = logit_scale * torch.einsum("...ic,...jc->...ij", l1, l2)  #
        logits_t = logits.transpose(-1, -2)
        labels = torch.arange(logits.size(-1), device=logits.device)
        if logits.ndim == 3:
            labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        loss_t = F.cross_entropy(logits_t, labels, reduction="mean")
        
        return loss, loss_t
    
    def content_loss(self, l, l_hat):
        if self.content_cosine:
            return self.cosine_distance_loss(l, l_hat)
        return F.mse_loss(l, l_hat, reduction="mean")
    
    def cross_entropy_loss(self, l, y, space):
        if space == 's':
            l = self.sub_clf_layer(l)
        elif space == 't':
            l = self.task_clf_layer(l)
        return F.cross_entropy(l, y, reduction="mean")
    
    def get_x(self, x):
        _, x1, s1, t1, r1 = self.loader.sample_batch(self.batch_size)
        x['x'] = x1.cuda()
        x['S'] = torch.tensor(s1-1, device='cuda')
        x['T'] = torch.tensor(t1, device='cuda')
        #x['R'] = r1.cuda()
        return x
    
    def get_l(self, x):
        if 'x' not in x:
            x = self.get_x(x)
        x['l'] = self.encode(x['x'])
        return x
    
    def get_x_hat(self, x):
        if 'l' not in x:
            x = self.get_l(x)
        x['x_hat'] = self.decode(x['l'])
        return x
    
    def get_l_hat(self, x):
        if 'x_hat' not in x:
            x = self.get_x_hat(x)
        x['l_hat'] = self.encode(x['x_hat'])
        return x
    
    def get_x_prop(self, x, prop='s'):
        if 'x_p_' + prop + '1' not in x or 'x_p_' + prop + '2' not in x:
            _, x1, s1, t1, r1 = self.loader.sample_by_property(prop)
            _, x2, s2, t2, r2 = self.loader.sample_by_property(prop)
            sample_size = x1.size(0)
            samples1 = [x1]
            samples2 = [x2]
            #s_s1 = [s1]
            #s_s2 = [s2]
            #t_s1 = [t1]
            #t_s2 = [t2]
            #r_s1 = [r1]
            #r_s2 = [r2]

            if self.loader.split not in ['test', 'dev', 'eval']:
                for i in range(0, self.batch_size // (2*sample_size) - 1):
                    _, x1, s1, t1, r1 = self.loader.sample_by_property(prop)
                    samples1.append(x1)
                    #s_s1.append(s1)
                    #t_s1.append(t1)
                    #r_s1.append(r1)
                    _, x2, s2, t2, r2 = self.loader.sample_by_property(prop)
                    samples2.append(x2)
                    #s_s2.append(s2)
                    #t_s2.append(t2)
                    #r_s2.append(r2)
            samples1 = torch.stack(samples1, 0)
            samples2 = torch.stack(samples2, 0)
            #s_s1 = np.stack(s_s1, 0)
            #s_s2 = np.stack(s_s2, 0)
            #t_s1 = np.stack(t_s1, 0)
            #t_s2 = np.stack(t_s2, 0)
            #r_s1 = np.stack(r_s1, 0)
            #r_s2 = np.stack(r_s2, 0)
            #x['S_p_' + prop + '1'] = s_s1
            #x['S_p_' + prop + '2'] = s_s2
            #x['T_p_' + prop + '1'] = t_s1
            #x['T_p_' + prop + '2'] = t_s2
            #x['R_p_' + prop + '1'] = r_s1
            #x['R_p_' + prop + '2'] = r_s2

            x['x_p_' + prop + '1'] = samples1.cuda()
            x['x_p_' + prop + '2'] = samples2.cuda()
        return x
    
    def collapsed_apply(self, x, f, *args, **kwargs):
        x_shape = x.size()
        x = x.view(-1, *x_shape[2:])
        x = f(x, *args, **kwargs)
        x = x.view(x_shape[0], x_shape[1], *x.size()[1:])
        return x
    
    def get_l_prop(self, x, prop='s'):
        raise NotImplementedError()
        if 'x_p_' + prop + '1' not in x or 'x_p_' + prop + '2' not in x:
            x = self.get_x_prop(x, prop)
        if 'l_p_' + prop + '1' not in x or 'l_p_' + prop + '2' not in x:
            x['l_p_' + prop + '1'] = self.collapsed_apply(x['x_p_' + prop + '1'], self.encode)
            x['l_p_' + prop + '2'] = self.collapsed_apply(x['x_p_' + prop + '2'], self.encode)
        return x
    
    def get_x_hat_prop(self, x, prop='s'):
        raise NotImplementedError()
        if 'l_p_' + prop + '1' not in x or 'l_p_' + prop + '2' not in x:
            x = self.get_l_prop(x, prop)
        if 'x_hat_p_' + prop + '1' not in x or 'x_hat_p_' + prop + '2' not in x:
            x['x_hat_p_' + prop + '1'] = self.collapsed_apply(x['l_p_' + prop + '1'], self.decode).view_as(x['x_p_' + prop + '1'])
            x['x_hat_p_' + prop + '2'] = self.collapsed_apply(x['l_p_' + prop + '2'], self.decode).view_as(x['x_p_' + prop + '2'])
        return x
    
    def determine_prop(self, k):
        prop = None
        if '_s' in k:
            prop = 's'
        elif '_t' in k:
            prop = 't'
        elif '_ru' in k:
            prop = 'ru'
        elif '_re' in k:
            prop = 're'
        else:
            raise ValueError("Unknown property")
        return prop
    
    def losses(self, x=None, which=["recon", "contra", "ortho", "content"], loader=None):
        raise NotImplementedError("Losses not implemented for this model")
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class SplitLatentModel(BaselineModel):
    
    def __init__(self, in_channels, channels, latent_dim, num_layers, kernel_size, recon_type=True, content_cosine=False):
        super(SplitLatentModel, self).__init__(in_channels, channels, latent_dim, num_layers, kernel_size, recon_type, content_cosine)
        
        self.encoder_out_net = ConvBlock(channels, channels, channels)
        
        self.subject_encoder_out_net = ConvBlock(channels, channels, latent_dim)
        
        self.subject_decoder_in_net = ConvBlock(latent_dim, channels, channels)
        
        self.task_encoder_out_net = ConvBlock(channels, channels, latent_dim)
        
        self.task_decoder_in_net = ConvBlock(latent_dim, channels, channels)
        
        self.decoder_mixin_net = ConvBlock(channels, channels, channels)
        
        self.recon_type = recon_type
        self.content_cosine = content_cosine
    
    def pre_encode(self, x):
        x = self.encoder_in(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.encoder_bottleneck(x)
        x = x.permute(0, 2, 1)
        return x
    
    def post_decode(self, x):
        x = x.permute(0, 2, 1)
        x = self.decoder_bottleneck(x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = self.decoder_out(x)
        return x
    
    def subject_task_encode(self, x):
        x = self.pre_encode(x)
        x = self.encoder_out_net(x)
        s = self.subject_encoder_out_net(x)
        t = self.task_encoder_out_net(x)
        s = s.view(-1, self.latent_seqs * self.latent_dim)
        t = t.view(-1, self.latent_seqs * self.latent_dim)
        return s, t
    
    def subject_task_decode(self, s, t):
        s = s.view(-1, self.latent_dim, self.latent_seqs)
        t = t.view(-1, self.latent_dim, self.latent_seqs)
        x = torch.cat([s, t], dim=1)
        x = self.decoder_in_net(x)
        s = self.subject_decoder_in_net(s)
        t = self.task_decoder_in_net(t)
        x_ = x + s + t
        x_ = self.decoder_mixin_net(x_)
        x = x + x_
        x = self.post_decode(x)
        return x
    
    def get_s_t(self, x):
        if 'x' not in x:
            x = self.get_x(x)
        if 's' not in x or 't' not in x:
            s, t = self.subject_task_encode(x['x'])
            x['s'] = s
            x['t'] = t
        return x
    
    def get_x_hat(self, x):
        if 's' not in x or 't' not in x:
            x = self.get_s_t(x)
        x['x_hat'] = self.subject_task_decode(x['s'], x['t'])
        return x
    
    def get_s_t_hat(self, x):
        if 'x_hat' not in x:
            x = self.get_x_hat(x)
        x['s_hat'], x['t_hat'] = self.subject_task_encode(x['x_hat'])
        return x
    
    def collapsed_split_apply(self, x, f, *args, **kwargs):
        x_shape = x.size()
        x = x.view(-1, *x_shape[2:])
        s, t = f(x, *args, **kwargs)
        s = s.view(x_shape[0], x_shape[1], *s.size()[1:])
        t = t.view(x_shape[0], x_shape[1], *t.size()[1:])
        return s, t
    
    def get_s_t_prop(self, x, prop):
        if 'x_p_' + prop + '1' not in x or 'x_p_' + prop + '2' not in x:
            x = self.get_x_prop(x, prop)
        if 's_p_' + prop + '1' not in x or 't_p_' + prop + '1' not in x or 's_p_' + prop + '2' not in x or 't_p_' + prop + '2' not in x:
            s1, t1 = self.collapsed_split_apply(x['x_p_' + prop + '1'], self.subject_task_encode)
            s2, t2 = self.collapsed_split_apply(x['x_p_' + prop + '2'], self.subject_task_encode)
            x['s_p_' + prop + '1'] = s1
            x['t_p_' + prop + '1'] = t1
            x['s_p_' + prop + '2'] = s2
            x['t_p_' + prop + '2'] = t2
        return x
    
    def get_x_hat_prop(self, x, prop):
        if 's_p_' + prop + '1' not in x or 't_p_' + prop + '1' not in x or 's_p_' + prop + '2' not in x or 't_p_' + prop + '2' not in x:
            x = self.get_s_t_prop(x, prop)
        x['x_hat_p_' + prop + '1'] = self.subject_task_decode(x['s_p_' + prop + '1'], x['t_p_' + prop + '1']).view_as(x['x_p_' + prop + '1'])
        x['x_hat_p_' + prop + '2'] = self.subject_task_decode(x['s_p_' + prop + '2'], x['t_p_' + prop + '2']).view_as(x['x_p_' + prop + '2'])
        return x
    
    def get_x_hat_prop_only_permuted(self, x, prop):
        if 's_p_' + prop + '1' not in x or 't_p_' + prop + '1' not in x or 's_p_' + prop + '2' not in x or 't_p_' + prop + '2' not in x:
            x = self.get_s_t_prop(x, prop)
        if 'ldetachswitch' in self.recon_type:
            if prop == 's':
                x['x_hat_p_' + prop + '_s1t2'] = self.subject_task_decode(x['s_p_' + prop + '1'].detach(), x['t_p_' + prop + '2']).view_as(x['x_p_' + prop + '1'])
                x['x_hat_p_' + prop + '_s2t1'] = self.subject_task_decode(x['s_p_' + prop + '2'].detach(), x['t_p_' + prop + '1']).view_as(x['x_p_' + prop + '2'])
            elif prop == 't':
                x['x_hat_p_' + prop + '_s1t2'] = self.subject_task_decode(x['s_p_' + prop + '1'], x['t_p_' + prop + '2'].detach()).view_as(x['x_p_' + prop + '1'])
                x['x_hat_p_' + prop + '_s2t1'] = self.subject_task_decode(x['s_p_' + prop + '2'], x['t_p_' + prop + '1'].detach()).view_as(x['x_p_' + prop + '2'])
        elif 'ldetach' in self.recon_type:
            if prop == 's':
                x['x_hat_p_' + prop + '_s1t2'] = self.subject_task_decode(x['s_p_' + prop + '1'], x['t_p_' + prop + '2'].detach()).view_as(x['x_p_' + prop + '1'])
                x['x_hat_p_' + prop + '_s2t1'] = self.subject_task_decode(x['s_p_' + prop + '2'], x['t_p_' + prop + '1'].detach()).view_as(x['x_p_' + prop + '2'])
            elif prop == 't':
                x['x_hat_p_' + prop + '_s1t2'] = self.subject_task_decode(x['s_p_' + prop + '1'].detach(), x['t_p_' + prop + '2']).view_as(x['x_p_' + prop + '1'])
                x['x_hat_p_' + prop + '_s2t1'] = self.subject_task_decode(x['s_p_' + prop + '2'].detach(), x['t_p_' + prop + '1']).view_as(x['x_p_' + prop + '2'])
        else:
            x['x_hat_p_' + prop + '_s1t2'] = self.subject_task_decode(x['s_p_' + prop + '1'], x['t_p_' + prop + '2']).view_as(x['x_p_' + prop + '1'])
            x['x_hat_p_' + prop + '_s2t1'] = self.subject_task_decode(x['s_p_' + prop + '2'], x['t_p_' + prop + '1']).view_as(x['x_p_' + prop + '2'])
        return x
    
    def get_s_t_hat_prop_any(self, x, prop):
        if 'x_hat_p_' + prop + '1' in x:
            s_hat_s1t1, t_hat_s1t1 = self.collapsed_split_apply(x['x_hat_p_' + prop + '1'], self.subject_task_encode)
            x['s_hat_p_' + prop + '1'] = s_hat_s1t1
            x['t_hat_p_' + prop + '1'] = t_hat_s1t1
        if 'x_hat_p_' + prop + '2' in x:
            s_hat_s2t2, t_hat_s2t2 = self.collapsed_split_apply(x['x_hat_p_' + prop + '2'], self.subject_task_encode)
            x['s_hat_p_' + prop + '2'] = s_hat_s2t2
            x['t_hat_p_' + prop + '2'] = t_hat_s2t2
        if 'x_hat_p_' + prop + '_s1t2' in x:
            s_hat_s1t2, t_hat_s1t2 = self.collapsed_split_apply(x['x_hat_p_' + prop + '_s1t2'], self.subject_task_encode)
            x['s_hat_p_' + prop + '_s1t2'] = s_hat_s1t2
            x['t_hat_p_' + prop + '_s1t2'] = t_hat_s1t2
        if 'x_hat_p_' + prop + '_s2t1' in x:
            s_hat_s2t1, t_hat_s2t1 = self.collapsed_split_apply(x['x_hat_p_' + prop + '_s2t1'], self.subject_task_encode)
            x['s_hat_p_' + prop + '_s2t1'] = s_hat_s2t1
            x['t_hat_p_' + prop + '_s2t1'] = t_hat_s2t1
        return x
    
    def get_x_hat_hat_restored(self, x, prop):
        if 'x_hat_p_' + prop + '_s2t1' not in x or 'x_hat_p_' + prop + '_s1t2' not in x:
            x = self.get_x_hat_prop_only_permuted(x, prop)
        if 's_hat_p_' + prop + '_s2t1' not in x or 't_hat_p_' + prop + '_s2t1' not in x or 's_hat_p_' + prop + '_s1t2' not in x or 't_hat_p_' + prop + '_s1t2' not in x:
            x = self.get_s_t_hat_prop_any(x, prop)
        if 'rswitch' in self.recon_type:
            x['x_hat_hat_p_' + prop + '1'] = self.subject_task_decode(x['s_hat_p_' + prop + '_s2t1'], x['t_hat_p_' + prop + '_s1t2']).view_as(x['x_hat_p_' + prop + '_s1t2'])
            x['x_hat_hat_p_' + prop + '2'] = self.subject_task_decode(x['s_hat_p_' + prop + '_s1t2'], x['t_hat_p_' + prop + '_s2t1']).view_as(x['x_hat_p_' + prop + '_s2t1'])
        else:
            x['x_hat_hat_p_' + prop + '2'] = self.subject_task_decode(x['s_hat_p_' + prop + '_s2t1'], x['t_hat_p_' + prop + '_s1t2']).view_as(x['x_hat_p_' + prop + '_s2t1'])
            x['x_hat_hat_p_' + prop + '1'] = self.subject_task_decode(x['s_hat_p_' + prop + '_s1t2'], x['t_hat_p_' + prop + '_s2t1']).view_as(x['x_hat_p_' + prop + '_s1t2'])
        return x
    
    #dummy gets
    def get_l(self, x):
        raise ValueError("Not Relevant")
    
    def get_l_prop(self, x, prop):
        raise ValueError("Not Relevant")
    
    def get_l_hat(self, x):
        raise ValueError("Not Relevant")
    
    def determine_space(self, k):
        space = None
        if k.startswith('sub'):
            space = 's'
        elif k.startswith('task'):
            space = 't'
        return space
    
    def losses(self, x=None, which=None, loader=None):
        old_loader = self.loader
        if loader is not None:
            self.loader = loader
        if which is None:
            which = self.used_losses
        if x is None:
            x = {}
        loss_dict = {}
        for k in which:
            space = self.determine_space(k)
            if space is None:
                if k.startswith('recon'):
                    if 'x_hat' not in x:
                        x = self.get_x_hat(x)
                    loss_dict[k] = self.reconstruction_loss(x['x'], x['x_hat'])
                elif k.startswith('cosine'):
                    if 'x_hat' not in x:
                        x = self.get_x_hat(x)
                    loss_dict[k] = self.cosine_distance_loss(x['x'], x['x_hat'])
                elif k.startswith('latent_permute'):
                    if self.loader is None:
                        raise ValueError("Need to pass loader to compute permute loss")
                    else:
                        prop = self.determine_prop(k)
                        if 'x_hat_p_' + prop + '_s1t2' not in x or 'x_hat_p_' + prop + '_s2t1' not in x:
                            x = self.get_x_hat_prop_only_permuted(x, prop)
                        if 'lswitch' in self.recon_type.lower():
                            if prop == 's':
                                loss_dict[k + '_1'] = self.reconstruction_loss(x['x_p_' + prop + '1'], x['x_hat_p_' + prop + '_s1t2'])
                                loss_dict[k + '_2'] = self.reconstruction_loss(x['x_p_' + prop + '2'], x['x_hat_p_' + prop + '_s2t1'])
                            elif prop == 't':
                                loss_dict[k + '_1'] = self.reconstruction_loss(x['x_p_' + prop + '1'], x['x_hat_p_' + prop + '_s2t1'])
                                loss_dict[k + '_2'] = self.reconstruction_loss(x['x_p_' + prop + '2'], x['x_hat_p_' + prop + '_s1t2'])
                        else:
                            if prop == 's':
                                loss_dict[k + '_1'] = self.reconstruction_loss(x['x_p_' + prop + '1'], x['x_hat_p_' + prop + '_s2t1'])
                                loss_dict[k + '_2'] = self.reconstruction_loss(x['x_p_' + prop + '2'], x['x_hat_p_' + prop + '_s1t2'])
                            elif prop == 't':
                                loss_dict[k + '_1'] = self.reconstruction_loss(x['x_p_' + prop + '1'], x['x_hat_p_' + prop + '_s1t2'])
                                loss_dict[k + '_2'] = self.reconstruction_loss(x['x_p_' + prop + '2'], x['x_hat_p_' + prop + '_s2t1'])
                        loss_dict[k] = 0.5*(loss_dict[k + '_1'] + loss_dict[k + '_2'])
                elif k.startswith('quadruplet_permute'):
                    if self.loader is None:
                        raise ValueError("Need to pass loader to compute permute loss")
                    elif self.loader.split != 'train':
                        pass
                    else:
                        if k.endswith('_f'):
                            Q_batch_size = self.batch_size
                        else:
                            Q_batch_size = self.batch_size // 4
                        subjects1 = np.random.choice(self.loader.unique_subjects, Q_batch_size, replace=True)
                        subjects2 = np.random.choice(self.loader.unique_subjects, Q_batch_size, replace=True)
                        tasks1 = np.random.choice(self.loader.unique_tasks, Q_batch_size, replace=True)
                        tasks2 = np.random.choice(self.loader.unique_tasks, Q_batch_size, replace=True)
                        all_subjects = np.concatenate([subjects1, subjects1, subjects2, subjects2])
                        all_tasks = np.concatenate([tasks1, tasks2, tasks1, tasks2])
                        all_x = self.loader.sample_by_condition(all_subjects, all_tasks).cuda()
                        all_s, all_t = self.subject_task_encode(all_x)
                        s11, t11 = all_s[:Q_batch_size], all_t[:Q_batch_size]
                        s12, t12 = all_s[Q_batch_size:2*Q_batch_size], all_t[Q_batch_size:2*Q_batch_size]
                        s21, t21 = all_s[2*Q_batch_size:3*Q_batch_size], all_t[2*Q_batch_size:3*Q_batch_size]
                        s22, t22 = all_s[3*Q_batch_size:], all_t[3*Q_batch_size:]
                        #x11 -> s12, t21
                        #x12 -> s11, t22
                        #x21 -> s22, t11
                        #x22 -> s21, t12
                        all_s_p = torch.cat([s12, s11, s22, s21], dim=0)
                        all_t_p = torch.cat([t21, t22, t11, t12], dim=0)
                        
                        all_x_hat = self.subject_task_decode(all_s_p, all_t_p)
                        loss_dict[k] = self.reconstruction_loss(all_x, all_x_hat)
                elif k.startswith('restored_permute'):
                    if self.loader is None:
                        raise ValueError("Need to pass loader to compute permute loss")
                    else:
                        prop = self.determine_prop(k)
                        if 'x_hat_hat_p_' + prop + '1' not in x or 'x_hat_hat_p_' + prop + '2' not in x:
                            x = self.get_x_hat_hat_restored(x, prop)
                        loss_dict[k + '_1'] = self.reconstruction_loss(x['x_p_' + prop + '1'], x['x_hat_hat_p_' + prop + '1'])
                        loss_dict[k + '_2'] = self.reconstruction_loss(x['x_p_' + prop + '2'], x['x_hat_hat_p_' + prop + '2'])
                        loss_dict[k] = 0.5*(loss_dict[k + '_1'] + loss_dict[k + '_2'])
                elif k.startswith('conversion_permute'):
                    if 's' not in x or 't' not in x:
                        x = self.get_s_t(x)
                    B = x['s'].shape[0]
                    s_perm = x['s'][torch.randperm(B, device=x['s'].device)]
                    t_perm = x['t'][torch.randperm(B, device=x['t'].device)]
                    x_hat1 = self.subject_task_decode(s_perm, x['t'])
                    x_hat2 = self.subject_task_decode(x['s'], t_perm)
                    loss_dict[k + '_1'] = self.reconstruction_loss(x['x'], x_hat1)
                    loss_dict[k + '_2'] = self.reconstruction_loss(x['x'], x_hat2)
                    loss_dict[k] = 0.5*(loss_dict[k + '_1'] + loss_dict[k + '_2'])
                elif k.startswith('scramble_permute'):
                    if 's' not in x or 't' not in x:
                        x = self.get_s_t(x)
                    B = x['s'].shape[0]
                    s_perm = x['s'][torch.randperm(B, device=x['s'].device)]
                    t_perm = x['t'][torch.randperm(B, device=x['t'].device)]
                    x_hat = self.subject_task_decode(s_perm, t_perm)
                    loss_dict[k] = self.reconstruction_loss(x['x'], x_hat)
                else:
                    self.loader = old_loader
                    raise ValueError("Unknown loss: " + k)
            else:
                if "contra" in k:
                    if self.loader is None:
                        raise ValueError("Need to pass loader to compute contrastive loss")
                    else:
                        prop = self.determine_prop(k)
                        if space + '_p_' + prop + '1' not in x or space + '_p_' + prop + '2' not in x:
                            x = self.get_s_t_prop(x, prop)
                        contr1, contr2 = self.contrastive_loss(x[space + '_p_' + prop + '1'], x[space + '_p_' + prop + '2'], which=space)
                        loss_dict[k + '_1'] = contr1
                        loss_dict[k + '_2'] = contr2
                        loss_dict[k] = 0.5*(contr1 + contr2)
                elif "content" in k:
                    if space not in x:
                        x = self.get_s_t(x)
                    if space + '_hat' not in x:
                        x = self.get_s_t_hat(x)
                    loss_dict[k] = self.content_loss(x[space], x[space + '_hat'])
                elif "cross" in k:
                    if space not in x:
                        x = self.get_s_t(x)
                    loss_dict[k] = self.cross_entropy_loss(x[space], x[space.upper()], space)
                else:
                    self.loader = old_loader
                    raise ValueError(f"Unknown loss {k}")
        self.loader = old_loader
        return x, loss_dict
    
    def forward(self, x):
        s, t = self.subject_task_encode(x)
        x_hat = self.subject_task_decode(s, t)
        return x_hat