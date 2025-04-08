import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import math
from torch.distributions.normal import Normal
from torch.distributions import Bernoulli
from .initializer import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def create_gaussian_kernel(num_channel, kernel_size, sigma):
    mean = (kernel_size - 1) / 2.0
    x_cord = torch.arange(kernel_size).view(1,kernel_size).repeat(num_channel,1).to(sigma)
    gaussian_kernel = (1./(2.*np.pi))*torch.exp(-(x_cord-mean)**2/(2*sigma**2) )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel,dim=1,keepdim=True)
    gaussian_kernel = gaussian_kernel.view(1, num_channel, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(num_channel, 1, 1)
    return gaussian_kernel.float()

class TransformerEncoderLayerWithHooks(TransformerEncoderLayer):
    def __init__(self,d_model, nhead, dim_feedforward, dropout):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fixup_initialization()

    def fixup_initialization(self):
        r"""
        http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        temp_state_dic = {}
        en_layers = 3

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
            elif name in ["self_attn.v_proj.weight",]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def attend(self, src, context_mask=None, **kwargs):
        attn_res = self.self_attn(src, src, src, attn_mask=context_mask, **kwargs)
        return (*attn_res, torch.tensor(0, device=src.device, dtype=torch.float))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            src: L, N, E (time x batch x neurons)
            weights: N, L, S (batch x target time x source time)
        """
        residual = src

        src2, weights, attention_cost = self.attend(
            src,
            context_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = residual + self.dropout1(src2)
        src = self.norm1(src)
        residual = src

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        src = self.norm2(src)

        return src

class RNN_Cell(nn.Module):
    def __init__(self, n_inp, n_hid, n_ch=1, act='tanh', ksize=3, init='eye', freeze_rnn='no', freeze_encoder='no',
                 solo_init='yes'):
        super(RNN_Cell, self).__init__()
        self.n_hid = n_hid
        self.n_ch = n_ch
        self.Wx = nn.Linear(n_inp, n_hid * n_ch)
        self.Wy = nn.Conv1d(n_ch, n_ch, ksize, padding=ksize // 2, padding_mode='circular')

        if solo_init == 'yes':
            nn.init.zeros_(self.Wx.weight)
            nn.init.zeros_(self.Wx.bias)
            with torch.no_grad():
                w = self.Wx.weight.view(n_ch, n_hid, n_inp)
                w[:, 0] = 1.0
        elif solo_init == 'no':
            nn.init.normal_(self.Wx.weight, mean=0.0, std=0.001)
        else:
            raise NotImplementedError

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'ident':
            self.act = nn.Identity()
        else:
            raise NotImplementedError

        assert init in ['eye', 'fwd', 'rand']

        if init == 'eye':
            wts = torch.zeros(n_ch, n_ch, ksize)
            nn.init.dirac_(wts)

        elif init == 'fwd':
            wts = torch.zeros(n_ch, n_ch, ksize)
            nn.init.dirac_(wts)
            wts = torch.roll(wts, 1, -1)

        if init == 'eye' or init == 'fwd':
            with torch.no_grad():
                self.Wy.weight.copy_(wts)

        if freeze_encoder == 'yes':
            for param in self.Wx.parameters():
                param.requires_grad = False
        else:
            assert freeze_encoder == 'no'

        if freeze_rnn == 'yes':
            for param in self.Wy.parameters():
                param.requires_grad = False
        else:
            assert freeze_rnn == 'no'

    def forward(self, x, hy):
        hy = self.act(self.Wx(x) + self.Wy(hy.view(-1, self.n_ch, self.n_hid)).flatten(start_dim=1))
        return hy


class KernelNormalizedLinear(nn.Linear):
    def forward(self, input):
        normed_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        return torch.nn.functional.linear(input, normed_weight, self.bias)

def pad_mask(mask, data, value):
    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return torch.nn.functional.pad(mask, pad_shape, value=value)

class CoordinatedDropout:
    def __init__(self, cd_rate):
        self.cd_rate = cd_rate
        self.cd_input_dist = Bernoulli(cd_rate)
        self.cd_pass_dist = Bernoulli(1.-cd_rate)

    def process_batch(self, batch):
        encod_data = batch
        # Only use CD where we are inferring rates (none inferred for IC segment)
        maskable_data = encod_data
        # Sample a new CD mask at each training step
        device = encod_data.device
        cd_mask = self.cd_input_dist.sample(maskable_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(maskable_data.shape).to(device)
        grad_mask = torch.logical_or(torch.logical_not(cd_mask), pass_mask).float()
        # Mask and scale post-CD input so it has the same sum as the original data
        cd_masked_data = maskable_data * cd_mask / self.cd_rate
        # Concatenate the data from the IC encoder segment if using
        cd_input = cd_masked_data

        return cd_input, grad_mask

    def process_losses(self, recon_loss, cd_mask):
        # Expand mask, but don't block gradients
        cd_mask = pad_mask(cd_mask, recon_loss, 1.0)
        # Block gradients with respect to the masked outputs
        grad_loss = recon_loss * cd_mask
        nograd_loss = (recon_loss * (1 - cd_mask)).detach()
        cd_loss = grad_loss + nograd_loss
        return cd_loss

class SampleValidation:
    def __init__(self, sv_rate, fwd_steps, heldin_neurons):
        self.sv_rate = sv_rate
        self.sv_input_dist = Bernoulli(sv_rate)
        self.heldin_neurons = heldin_neurons
        self.fwd_steps = fwd_steps

    def process_batch(self, batch):

        unmasked_data1 = batch[:,:-self.fwd_steps,:]
        unmasked_data2 = batch[:,-self.fwd_steps:,:self.heldin_neurons]
        masked_data = batch[:,-self.fwd_steps:,self.heldin_neurons:]

        device = batch.device
        sv_mask = self.sv_input_dist.sample(masked_data.shape).to(device)
        pass_mask1 = torch.ones_like(unmasked_data1).to(device)
        pass_mask2 = torch.ones_like(unmasked_data2).to(device)
        masked_data = masked_data * sv_mask / self.sv_rate

        sv_data = torch.cat([unmasked_data2,masked_data],dim=2)
        sv_data = torch.cat([unmasked_data1,sv_data], dim=1)
        sv_mask = torch.cat([pass_mask2,sv_mask],dim=2)
        sv_mask = torch.cat([pass_mask1,sv_mask],dim=1)

        return sv_data, sv_mask

    def process_losses(self, recon_loss, sv_mask):
        grad_loss = recon_loss * sv_mask
        nograd_loss = (recon_loss * (1 - sv_mask)).detach()
        sv_loss = grad_loss + nograd_loss
        return sv_loss

class Potential(nn.Module):
    def __init__(self, z_in: int, x_in: int, dropout: float):
        super(Potential,self).__init__()

        self.conv_z = nn.Parameter(torch.zeros(4,4,3,requires_grad=True))
        #self.layer_x = nn.Sequential(
        #    KernelNormalizedLinear(x_in, z_in),
        #    nn.ReLU(True),
        #    KernelNormalizedLinear(z_in, z_in),
        #)

    def forward(self, z, x):
        conv_z = torch.nn.functional.normalize(self.conv_z, p=2, dim=2)
        z = z.reshape(z.size(0), 4, -1)
        Vz = torch.nn.functional.conv1d(input=z,weight=conv_z,bias=None,stride=1,padding=1)
        Vz = Vz.bmm(z.transpose(1, 2))
        return Vz.sum(-1) #+ (x*z).sum(-1)


class LangevinAutoencoder(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            fwd_steps: int,
            learning_rate: float,
            weight_decay: float,
            dropout: float,
            gamma: float,
            cd_rate: float,
    ):
        """Initializes the model.

        Parameters
        ----------
        input_size : int
            The dimensionality of the input sequence (i.e.
            number of heldin neurons)
        hidden_size : int
            The hidden dimensionality of the network, which
            determines the dimensionality of both the encoders
            and decoders
        output_size : int
            The dimensionality of the output sequence (i.e.
            total number of heldin and heldout neurons)
        fwd_steps: int
            The number of time steps to unroll beyond T
        learning_rate : float
            The learning rate to use for optimization
        weight_decay : float
            The weight decay to regularize optimization
        dropout : float
            The ratio of neurons to drop in dropout layers
        """
        super().__init__()
        self.save_hyperparameters()

        # Instantiate GRU encoder
        self.encoder = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
        )
        #RNN_Cell(n_inp=input_size,n_hid=hidden_size//7, n_ch=7)
        #nn.GRUCell(
        #    input_size=input_size,
        #    hidden_size=hidden_size,
        #    bias=True,
        #)
        #RNN_Cell(n_inp=input_size,n_hid=hidden_size)
        #ClippedGRUCell(
        #    input_size=input_size,
        #    hidden_size=hidden_size,
        #    clip_value = 200,
        #    is_encoder= True
        #)

        # Instantiate linear mapping to initial Gaussians
        self.linear_z_means = nn.Linear(hidden_size, hidden_size)
        self.linear_z_logvar = nn.Linear(hidden_size, hidden_size)

        self.linear_v_means = nn.Linear(hidden_size, hidden_size)
        self.linear_v_logvar = nn.Linear(hidden_size, hidden_size)

        # Instantiate GRU encoder
        #self.decoder_z = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size//2),
        #    nn.ReLU(True),
        #)
        #self.decoder_v = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size//2),
        #    nn.ReLU(True),
        #)
        #self.decoder_hid = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(True),
        #)
        self.decoder = TransformerEncoderLayerWithHooks(d_model=hidden_size*3, nhead=2, dim_feedforward=512,dropout=dropout)
        #self.decoder = TransformerEncoder(self.decoder_layer, num_layers=3)#TransformerEncoderLayer() #TransformerEncoderLayerWithHooks(d_model=hidden_size*3,nhead=8,dim_feedforward=512,dropout=dropout)
        # Instantiate Potential
        self.potential = Potential(z_in=hidden_size,x_in=input_size, dropout=dropout)
        # Instantiate linear readout
        self.readout = nn.Linear(
            in_features=3*hidden_size,
            out_features=output_size,
        )
        # Instantiate dropout
        self.dropout = nn.Dropout(p=dropout)
        #Damping rato of Langevin eq
        self.gamma = gamma
        self.step = 0.01
        #Prior diffusion rates
        #self.D = nn.Parameter(torch.ones(1, 1, requires_grad=True))
        #linear model parameters
        #self.regress_models = nn.Parameter(torch.zeros((1,11,output_size), requires_grad=True))
        #Coordinated Dropout
        self.cd = CoordinatedDropout(cd_rate=cd_rate)
        #Sample Validation
        self.sv = SampleValidation(sv_rate=0.8,fwd_steps=fwd_steps,heldin_neurons=input_size)
        self.switch_epoch_l2 = 500.
        self.switch_epoch_kl = 500.
        #self.enc_h0 = nn.Parameter(torch.zeros((1, hidden_size), requires_grad=True))

    def reparameterize(self, mu, log_var, logvar=True):

        if logvar==True:
            std = torch.exp(0.5 * log_var)
        else:
            std = log_var
        eps = torch.randn_like(std)

        return mu + eps * std

    def kl_gauss(self, mean, log_var, mean2=0.0, var2=1.0):
        #KLD = -0.5 * torch.sum(1 + log_var - (mean-mean2).pow(2) - log_var.exp())
        KLD = 0.5 * torch.sum(math.log(var2) - log_var - 1 + ((mean - mean2) ** 2) / var2 + log_var.exp() / var2)
        return KLD / mean.size(0)

    def kl_two_gauss(self, mean1, var1, mean2, var2):

        KLD = 0.5 * torch.sum(torch.log(var2) - torch.log(var1) -1 + ((mean2-mean1)**2)/var2 + var1/var2)

        return KLD / mean1.size(0)

    def forward(self, observ, use_logrates=True):
        """The forward pass of the model.

                Parameters
                ----------
                observ : torch.Tensor
                    A BxTxN tensor of heldin neurons at observed
                    time points.
                use_logrates: bool
                    Whether to output logrates for training
                    or firing rates for analysis.

                Returns
                -------
                torch.Tensor
                    A Bx(T+fwd_steps)x(N+n_heldout) tensor of
                    estimated firing rates
                torch.Tensor
                    A Bx(T+fwd_steps)x(hidden_dim) tensor of
                    latent states
                """
        batch_size, obs_steps, num_heldin = observ.shape
        hidden = self.encoder(observ[:, 0])
        hidden = self.dropout(hidden)
        #hidden = hidden.view(hidden.size(0), -1)
        z_mu, z_logvar = self.linear_z_means(hidden), self.linear_z_logvar(hidden)
        v_mu, v_logvar = self.linear_v_means(hidden), self.linear_v_logvar(hidden)
        z, v = self.reparameterize(z_mu,z_logvar), self.reparameterize(v_mu,v_logvar)
        #hidden_de = torch.cat([self.decoder_z(z), self.decoder_v(v), self.decoder_hid(hidden)], dim=1)
        #hidden_de = self.dropout(hidden_de)
        #hidden_de = torch.cat([self.decoder_z(z), self.decoder_v(v), self.decoder_hid(hidden)], dim=1)
        #logrates = self.readout(hidden_de)
        #logrates = logrates.unsqueeze(1)
        latents = torch.cat([z,v,hidden],dim=1).unsqueeze(1)
        #Intial KL Loss
        KL_loss = self.kl_gauss(z_mu,z_logvar,mean2=0.0,var2=1.0) + self.kl_gauss(v_mu,v_logvar,mean2=0.0,var2=1.0)
        for t in range(1,obs_steps + self.hparams.fwd_steps):
            if t<obs_steps:
                hidden = self.encoder(observ[:,t-1], hidden)
                hidden = self.dropout(hidden)
                #hidden = hidden.view(hidden.size(0), -1)
                input = observ[:,t]
            else:
                #spike_lst = 0
                #samples = 50
                #for i in range(samples):
                #    spike_lst += torch.poisson(torch.exp(logrates[:,-1,:num_heldin]))
                #spike_lst = spike_lst/samples
                # A simple linear model predict r_i+1 give history 10 points
                #lograte_t1 = logrates[:, -1, :num_heldin] + torch.mean(
                #    logrates[:, -10:, :num_heldin] - logrates[:, -11:-1, :num_heldin], dim=1)
                #spike_fwd = 0
                #for i in range(samples):
                #    spike_fwd += torch.poisson(torch.exp(lograte_t1))
                #spike_fwd = spike_fwd / samples
                hidden = self.encoder(observ[:,-1], hidden)
                hidden = self.dropout(hidden)
                #hidden = hidden.view(hidden.size(0), -1)
                input = observ[:,-1] #spike_fwd
            z = z.clone().requires_grad_()
            U = self.potential(z, input)
            u_z = grad(U.sum(), z, create_graph=True)[0]
            #Hamiltonian Flow
            z = z + self.step * v
            v = v - self.step * u_z
            #Probabilistic Step
            v = self.reparameterize((1-self.gamma)*v, math.sqrt(2 * self.gamma)*torch.ones_like(v), logvar=False)
            #Decode [Z,V,H]
            #hidden_de = torch.cat([self.decoder_z(z), self.decoder_v(v),self.decoder_hid(hidden)], dim=1)
            #hidden_de = self.decoder_hid(hidden)
            #hidden_de = self.dropout(hidden_de)
            #lograte_t = self.readout(hidden_de)
            #logrates = torch.cat([logrates,lograte_t.unsqueeze(1)],dim=1)
            latents = torch.cat([latents, torch.cat([z,v,hidden],dim=1).unsqueeze(1)], dim=1)
            KL_loss += self.kl_gauss(v,2 * self.gamma * torch.ones_like(v), mean2=0.0, var2=0.1)#2 * self.D * t + 1.0)
        logrates = self.decoder(latents.transpose(0,1))
        logrates = self.readout(logrates)
        logrates = logrates.transpose(0,1)
        if use_logrates:
            return logrates, latents, KL_loss
        else:
            return torch.exp(logrates), latents, KL_loss

    def on_before_optimizer_step(
            self, optimizer: torch.optim.Optimizer, optimizer_idx: int
    ):
        """
        This method is called before each optimizer step to gradually ramp up weight decay.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer that will be used to update the model's parameters.
        optimizer_idx : int
            The index of the optimizer.
        """

        # Gradually ramp weight decay alongside the l2 parameters
        l2_ramp = (self.current_epoch) / (self.switch_epoch_l2)
        l2_ramp = torch.clamp(torch.tensor(l2_ramp), 0, 1)
        optimizer.param_groups[0]["weight_decay"] = l2_ramp * self.hparams.weight_decay

    def configure_optimizers(self):
        """Sets up the optimizer.

        Returns
        -------
        torch.optim.Adam
            A configured optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.95,
            patience=10,
            threshold=0.0,
            min_lr=1e-5,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "hp_metric",
        }
        #return optimizer
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        #return [optimizer], [scheduler]

    def training_step(self, batch, batch_ix):
        """Computes, logs, and returns the loss.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch of data from the datamodule - contains
            heldin, heldin_forward, heldout, heldout_forward,
            and behavior tensors.
        batch_ix : int
            Ignored

        Returns
        -------
        torch.Tensor
            The scalar loss
        """

        input_data, recon_data, behavior = batch

        cd_data, cd_mask = self.cd.process_batch(input_data)
        #sv_data, sv_mask = self.sv.process_batch(recon_data)

        # Pass data through the model
        #preds, _, KL_loss = self.forward(input_data, use_logrates=True)
        cd_preds, _, cd_KL_loss = self.forward(cd_data, use_logrates=True)
        # Compute the Poisson log-likelihood
        kl_ramp = (self.current_epoch) / (self.switch_epoch_kl)
        kl_ramp = 1e-1*torch.clamp(torch.tensor(kl_ramp), 0, 1)

        cd_nll_loss = nn.functional.poisson_nll_loss(cd_preds, recon_data, reduction='none')
        #sv_nll_loss = nn.functional.poisson_nll_loss(preds, sv_data, reduction='none')

        cd_nll_loss = self.cd.process_losses(cd_nll_loss, cd_mask)
        #sv_nll_loss = self.sv.process_losses(sv_nll_loss, sv_mask)
        #nll_loss = nn.functional.poisson_nll_loss(preds, recon_data)
        loss =  cd_nll_loss.mean() + kl_ramp*(cd_KL_loss)

        self.log("train/loss", loss)

        return loss

    #def linear_fit(self, logrates):
    #    for t in range(logrates.size(1)):
    #        if t < 10:
    #            logrates_ar_t = logrates[:, t] + torch.sum(
    #                self.regress_models[:, 1:t] * torch.poisson(torch.exp(logrates[:, 1:t])),
    #                dim=1) + self.regress_models[:, 0]
    #            if t == 0:
    #                logrates_ar = logrates_ar_t.unsqueeze(1)
    #            else:
    #                logrates_ar = torch.cat([logrates_ar, logrates_ar_t],dim=1)
    #        else:
    #            logrates_ar_t = logrates[:, t] + torch.sum(
    #                self.regress_models[:, 1:] * torch.poisson(torch.exp(logrates[:, t-10:t])),
    #                dim=1) + self.regress_models[:, 0]
    #            logrates_ar = torch.cat([logrates_ar, logrates_ar_t], dim=1)
    #    return logrates_ar

    def validation_step(self, batch, batch_ix):
        """Computes, logs, and returns the loss.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A batch of data from the datamodule. During the
            "val" phase, contains heldin, heldin_forward,
            heldout, heldout_forward, and behavior tensors.
            During the "test" phase, contains only the heldin
            tensor.
        batch_ix : int
            Ignored

        Returns
        -------
        torch.Tensor
            The scalar loss
        """
        torch.set_grad_enabled(True)
        # On test-phase data, compute loss only across heldin neurons
        if len(batch) == 1:
            (input_data,) = batch
            # Pass data through the model
            preds, latents, KL_loss = self.forward(input_data, use_logrates=True)
            # Isolate heldin predictions
            _, n_obs, n_heldin = input_data.shape
            preds = preds[:, :n_obs, :n_heldin]
            recon_data = input_data
        else:
            input_data, recon_data, behavior = batch
            # Pass data through the model
            preds, latents, KL_loss = self.forward(input_data, use_logrates=True)
        # Compute the Poisson log-likelihood
        sv_data, sv_mask = self.sv.process_batch(recon_data)
        loss = nn.functional.poisson_nll_loss(preds, sv_data, reduction='none')
        loss = self.sv.process_losses(loss, sv_mask)
        #loss = nn.functional.poisson_nll_loss(preds, recon_data)
        self.log("valid/loss", loss)
        self.log("hp_metric", loss)
        self.log("damping", self.gamma)
        return loss
