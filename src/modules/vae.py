import torch
import torch.nn as nn

class VAE(nn.Module):
    name = 'vae'
    def __init__(self, var_coef=1.0, eval_vae=False):
        """
        Parameters
        ----------
        var_coef: float
        eval_vae: bool        
        """
        super().__init__()
        self.var_coef = var_coef
        self.eval_vae = eval_vae
        self._device_param = nn.Parameter(torch.zeros((0,)))

    @property
    def device(self):
        return self._device_param.device
    
    def forward(self, mode='train', mu=None, var=None, latent_size=None, batch_size=None):
        """
        Parameters
        ----------
        mode: Either 'train', 'eval' or 'generate'
        """
        if mode == 'generate':
            return torch.randn(size=(batch_size, latent_size), device=self.device)
        else:
            if mode == 'train' or self.eval_vae:
                latent = mu + torch.randn(*mu.shape, device=mu.device)*torch.sqrt(var)*self.var_coef
            else:
                latent = mu
            return latent
class MinusD_KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, var):
        return 0.5*(torch.sum(mu**2)+torch.sum(var)-torch.sum(torch.log(var))-var.numel())
    