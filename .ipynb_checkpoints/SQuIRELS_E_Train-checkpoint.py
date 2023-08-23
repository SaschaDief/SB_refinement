import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor


num_steps = 20
gamma_max = 0.001
gamma_min = 0.001


suffix = '_GFlash_Energy'

CUDA = True
device = torch.device("cuda" if CUDA else "cpu")



### https://www.zijianhu.com/post/pytorch/ema/
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)







file_path_gflash = './data/run_GFlash01_100k_10_100GeV_full'
file_path_g4 = './data/run_Geant_100k_10_100GeV_full'
file_name = '.npy'

energy_voxel_g4 = np.load(file_path_g4 + file_name)[:, 0:100].astype(np.float32)
energy_voxel_gflash  = np.load(file_path_gflash + file_name)[:, 0:100].astype(np.float32)

energy_particle_g4 = np.load(file_path_g4 + file_name)[:, 200:201].astype(np.float32)/10000.0
energy_particle_gflash  = np.load(file_path_gflash + file_name)[:, 200:201].astype(np.float32)/10000.0


# sort by incident energy to define pairs
mask_energy_particle_g4 = np.argsort(energy_particle_g4, axis=0)[:,0]
mask_energy_particle_gflash = np.argsort(energy_particle_gflash, axis=0)[:,0]

energy_particle_g4 = energy_particle_g4[mask_energy_particle_g4]
energy_particle_gflash = energy_particle_gflash[mask_energy_particle_gflash]

energy_voxel_g4 = energy_voxel_g4[mask_energy_particle_g4]
energy_voxel_gflash = energy_voxel_gflash[mask_energy_particle_gflash]


# reshuffle consistently
mask_shuffle = np.random.permutation(energy_particle_g4.shape[0])

energy_particle_g4 = energy_particle_g4[mask_shuffle]
energy_particle_gflash = energy_particle_gflash[mask_shuffle]

energy_voxel_g4 = energy_voxel_g4[mask_shuffle]
energy_voxel_gflash = energy_voxel_gflash[mask_shuffle]


energy_g4 = np.sum(energy_voxel_g4, 1, keepdims=True)
energy_gflash = np.sum(energy_voxel_gflash, 1, keepdims=True)


energy_voxel_g4 = np.reshape(energy_voxel_g4, (-1, 1, 10, 10))
energy_voxel_gflash = np.reshape(energy_voxel_gflash, (-1, 1, 10, 10))


energy_voxel_g4 = energy_voxel_g4/np.tile(np.reshape(energy_g4, (-1, 1, 1, 1)), (1, 1, 10, 10))
energy_voxel_gflash = energy_voxel_gflash/np.tile(np.reshape(energy_gflash, (-1, 1, 1, 1)), (1, 1, 10, 10))


energy_g4 = energy_g4/energy_particle_g4
energy_gflash = energy_gflash/energy_particle_gflash



shifter_g4 = np.mean(energy_voxel_g4, 0)
shifter_gflash = np.mean(energy_voxel_gflash, 0)
scaler_g4 = np.std(energy_voxel_g4, 0)
scaler_gflash = np.std(energy_voxel_gflash, 0)

energy_voxel_g4 = (energy_voxel_g4 - shifter_g4)/scaler_g4
energy_voxel_gflash = (energy_voxel_gflash - shifter_gflash)/scaler_gflash



shifter_energy_g4 = np.mean(energy_g4, 0)
shifter_energy_gflash = np.mean(energy_gflash, 0)
scaler_energy_g4 = np.std(energy_g4, 0)
scaler_energy_gflash = np.std(energy_gflash, 0)

energy_g4 = (energy_g4 - shifter_energy_g4)/scaler_energy_g4
energy_gflash = (energy_gflash - shifter_energy_gflash)/scaler_energy_gflash




batch_size = 1024*16


npar = int(energy_voxel_g4.shape[0])

            
X_init = energy_gflash
Y_init = energy_particle_gflash
init_sample = torch.tensor(X_init)#.view(X_init.shape[0], 1, 10, 10)
init_lable = torch.tensor(Y_init)
init_ds = TensorDataset(init_sample, init_lable)
init_dl = DataLoader(init_ds, batch_size=batch_size, shuffle=False)
#init_dl = repeater(init_dl)
print(init_sample.shape)
            



X_final = energy_g4
Y_final = energy_particle_g4
final_sample = torch.tensor(X_final)#.view(X_final.shape[0], 1, 10, 10)
final_label = torch.tensor(Y_final)
final_ds = TensorDataset(final_sample, final_label)
final_dl = DataLoader(final_ds, batch_size=batch_size, shuffle=False)
#final_dl = repeater(final_dl)

#mean_final = torch.tensor(0.)
#var_final = torch.tensor(1.*10**3) #infty like

mean_final = torch.zeros(final_sample.size(-1)).to(device)
var_final = 1.*torch.ones(final_sample.size(-1)).to(device)

print(final_sample.shape)
print(mean_final.shape)
print(var_final.shape)


dls = {'f': init_dl, 'b': final_dl}





import torch
import torch.nn.functional as F
from torch import nn
import math


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb





class ScoreNetwork(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = MLP(3 * t_enc_dim + 1,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
            
        self.e_encoder = MLP(1,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out
        
        


lr = 1e-5

n = num_steps//2
gamma_half = np.linspace(gamma_min, gamma_max, n)
gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
gammas = torch.tensor(gammas).to(device)
T = torch.sum(gammas)

print(gammas)



## decay=1.0: No change on update
## decay=0.0: No memory of previous updates, memory is euqal to last update
## decay=0.9: New value 9 parts previous updates, 1 part current update
## decay=0.95: New value 49 parts previous updates, 1 part current update

model_f = ScoreNetwork(n_cond = init_lable.size(1)).to(device)
model_b = ScoreNetwork(n_cond = init_lable.size(1)).to(device)

model_f = torch.nn.DataParallel(model_f)
model_b = torch.nn.DataParallel(model_b)

opt_f = torch.optim.Adam(model_f.parameters(), lr=lr)
opt_b = torch.optim.Adam(model_b.parameters(), lr=lr)

net_f = EMA(model=model_f, decay=0.95).to(device)
net_b = EMA(model=model_b, decay=0.95).to(device)

nets  = {'f': net_f, 'b': net_b }
opts  = {'f': opt_f, 'b': opt_b }

nets['f'].train()
nets['b'].train()


d = init_sample[0].shape  # shape of object to diffuse
dy = init_lable[0].shape  # shape of object to diffuse
print(d)
print(dy)

#print(net_f)


def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout


def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + \
        torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout




class CacheLoader(Dataset):
    def __init__(self, forward_or_backward = 'f', forward_or_backward_rev = 'b', first = False, sample = False):
        super().__init__()
        self.num_batches = int(npar/batch_size)

        self.data = torch.zeros((self.num_batches, batch_size*num_steps, 2, *d)).to(device)  # .cpu()
        self.y_data = torch.zeros((self.num_batches, batch_size*num_steps, *dy)).to(device)  # .cpu()
        self.steps_data = torch.zeros((self.num_batches, batch_size*num_steps, 1)).to(device)  # .cpu() # steps



        for b, dat in enumerate(dls[forward_or_backward]):    
            #print(b, self.num_batches)
            
            if b == self.num_batches:
                break

            x = dat[0].float().to(device)
            x_orig = x.clone().to(device)
            y = dat[1].float().to(device)
            steps = torch.arange(num_steps).to(device)
            time = torch.cumsum(gammas, 0).to(device).float()


            N = x.shape[0]
            steps = steps.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            time = time.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            #gammas_new = gammas.reshape((1, num_steps, 1)).repeat((N, 1, 1))
            steps = time

            x_tot = torch.Tensor(N, num_steps, *d).to(x.device)
            y_tot = torch.Tensor(N, num_steps, *dy).to(y.device)
            out = torch.Tensor(N, num_steps, *d).to(x.device)
            store_steps = steps
            num_iter = num_steps
            steps_expanded = time

            with torch.no_grad():
                if first:

                    for k in range(num_iter):
                        gamma = gammas[k]
                        gradx = grad_gauss(x, mean_final, var_final)

                        t_old = x + gamma * gradx
                        z = torch.randn(x.shape, device=x.device)
                        x = t_old + torch.sqrt(2 * gamma)*z
                        gradx = grad_gauss(x, mean_final, var_final)

                        t_new = x + gamma * gradx

                        x_tot[:, k, :] = x
                        y_tot[:, k, :] = y

                        out[:, k, :] = (t_old - t_new)  # / (2 * gamma)


                else:
                    for k in range(num_iter):
                        gamma = gammas[k]
                        t_old = x + nets[forward_or_backward_rev](x, steps[:, k, :], y, x_orig)

                        if sample & (k == num_iter-1):
                            x = t_old
                        else:
                            z = torch.randn(x.shape, device=x.device)
                            x = t_old + torch.sqrt(2 * gamma) * z
                        t_new = x + nets[forward_or_backward_rev](x, steps[:, k, :], y, x_orig)

                        x_tot[:, k, :] = x
                        y_tot[:, k, :] = y
                        
                        
                        out[:, k, :] = (t_old - t_new)

                x_tot = x_tot.unsqueeze(2)
                out = out.unsqueeze(2)

                batch_data = torch.cat((x_tot, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data
                
                
                y_tot = y_tot.unsqueeze(1)
                
                flat_y_data = y_tot.flatten(start_dim=0, end_dim=1)
                self.y_data[b] = flat_y_data.flatten(start_dim=0, end_dim=1)


                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.y_data = self.y_data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        print('Cache size: {0}'.format(self.data.shape))

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        y = self.y_data[index]
        
        return x, out, y, steps

    def __len__(self):
        return self.data.shape[0]        
        
                    
                    

                    
def iterate_ipf(n_iter = 200, forward_or_backward = 'f', forward_or_backward_rev = 'b', first = False, sample = False):
                    
    CL = CacheLoader(forward_or_backward, forward_or_backward_rev, first, sample)
    CL = DataLoader(CL, batch_size=batch_size, shuffle=False)

    for i_iter in range(n_iter):

        for (i, data_iter) in enumerate(CL):
            (x, out, y, steps_expanded) = data_iter
            x = x.to(device)
            x_orig = x.clone().to(device)
            y = y.to(device)
            out = out.to(device)
            steps_expanded = steps_expanded.to(device)
            eval_steps = T - steps_expanded


            pred = nets[forward_or_backward](x, eval_steps, y, x_orig)

            loss = F.mse_loss(pred, out)
            loss.backward()
    
            #print(loss)
    
            opts[forward_or_backward].step()
            opts[forward_or_backward].zero_grad()
            
        print(loss)
        #EMA update
        nets[forward_or_backward].update()

    
start_iter=0

for i in range(1, 100):
    try:
        nets['f'].load_state_dict(torch.load('./models/Iter{:d}_net_f'.format(i) + suffix + '.pth', map_location=device))
        nets['b'].load_state_dict(torch.load('./models/Iter{:d}_net_b'.format(i) + suffix + '.pth', map_location=device))
        
        start_iter = i
    except:
        continue

n_iter_glob = 50

if start_iter == 0:  
    iterate_ipf(n_iter = 100, forward_or_backward = 'f', forward_or_backward_rev = 'b', first = True)

    
nets['f'].train()
nets['b'].train()

for i in range(start_iter+1, start_iter+20):

    iterate_ipf(n_iter = n_iter_glob, forward_or_backward = 'b', forward_or_backward_rev = 'f', first = False)
    iterate_ipf(n_iter = n_iter_glob, forward_or_backward = 'f', forward_or_backward_rev = 'b', first = False)

    torch.save(net_f.state_dict(), './models/Iter{:d}_net_f'.format(i) + suffix + '.pth')
    torch.save(net_b.state_dict(), './models/Iter{:d}_net_b'.format(i) + suffix + '.pth')


