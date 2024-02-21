import torch
import torch.nn as nn

class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, padding_mask):
        """
        Parameters
        ----------
        input: torch.tensor of torch.float [length, batch_size, feature_size]
        padding_mask: torch.tensor of torch.bool [length, batch_size]
            Pisitions where padding_mask is True is regarded as pad_token and ignored.
        """
        padding_mask = ~padding_mask.unsqueeze(-1)
        return torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)

class StartPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input[0]
        
class MaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask):
        masked_max_input = input.masked_fill(padding_mask.unsqueeze(-1), -torch.inf)
        return torch.max(masked_max_input, dim=0)[0]

class MeanStartMaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = ~padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(~padding_mask, -torch.inf)
        return torch.cat([
            torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0),
            input[self.slice],
            torch.max(masked_max_input, dim=0)[0]], dim=-1)
    
class MeanStartEndMaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor):
        padding_mask = padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(padding_mask, -torch.inf)
        padding_mask = ~padding_mask
        end_mask = end_mask.unsqueeze(-1)
        return torch.cat([
            torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0),
            input[0], 
            torch.sum(input*end_mask, dim=0),
            torch.max(masked_max_input, dim=0)[0]], dim=-1)
    
class MeanStdStartEndMaxMinPooler(nn.Module):
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor):
        padding_mask = padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(padding_mask, -torch.inf)
        masked_min_input = input.masked_fill(padding_mask, torch.inf)
        padding_mask = ~padding_mask
        end_mask = end_mask.unsqueeze(-1)
        mean = torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)
        std = torch.sum(((input - mean.unsqueeze(0))**2)*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)
        return torch.cat([
            mean,
            std,
            input[0], 
            torch.sum(input*end_mask, dim=0),
            torch.max(masked_max_input, dim=0)[0],
            torch.min(masked_min_input, dim=0)[0]], dim=-1)

class NoAffinePooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.start_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.max_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input, padding_mask):
        padding_mask = ~padding_mask
        masked_max_input = input + torch.log(padding_mask).unsqueeze(-1)
        return torch.cat([
            self.mean_norm(torch.sum(input*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1)),
            self.start_norm(input[0]),
            self.max_norm(torch.max(masked_max_input, dim=0)[0])], dim=-1)
