import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..models import function_name2func, function_config2func, \
    init_config2func

# sequence modules
class TeacherForcer(nn.Module):
    def __init__(self, length_dim):
        super().__init__()
        self.length_dim = length_dim
        self.input_slices = [slice(None)]*length_dim+[slice(None, -1)]
        self.target_slices = [slice(None)]*length_dim+[slice(1, None)]
    def forward(self, input, return_len=False):
        """
        Parameters
        ----------
        input: (any)[..., legnth, ...]
        return_len(bool):
        
        Returns
        -------
        input: (any)[..., length-1, ...]
        target: [..., length-1, ...]
        """
        return_ = input[self.input_slices], input[self.target_slices]
        if return_len:
            return_ += (return_[-1].shape[self.length_dim], )
        return return_

class MaskMaker(nn.Module):
    def __init__(self, mask_token, dtype='bool', direction='equal'):
        super().__init__()
        self.mask_token = mask_token
        self.dtype = dtype
        self.direction = direction
    def forward(self, input: torch.Tensor):
        """
        Parameters
        ----------
        input: (torch.int or long)[...]

        Returns
        -------
        mask: (torch.bool or int)[...]
        """
        if self.direction == 'equal':
            mask = input == self.mask_token
        else:
            mask = input != self.mask_token
        if self.dtype == 'bool':
            pass
        elif self.dtype == 'int':
            mask = mask.to(torch.int)
        return mask

class SelfAttentionLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, activation, d_ff_factor=None, dim_feedforward=None, norm_first=True, **kwargs):
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(f"Please specify either 'dim_feedforward'({dim_feedforward})"
                +f" XOR 'd_ff_factor'({d_ff_factor})")
        if dim_feedforward is None:
            dim_feedforward = int(d_model*d_ff_factor)
        activation = function_name2func[activation]
        super().__init__(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, norm_first=norm_first, **kwargs)

def get_posenc(length: int, emb_size: int) -> torch.Tensor:
    """
    Returns
    -------
    pe: torch.tensor(float)[length, 1(batch_size dim), emb_size]
    
    """
    pe = torch.zeros(length, emb_size)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_size, 2) *
                            -(math.log(10000.0) / emb_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1)
    return pe

def load_pe_pre_hook_load(model, state_dict, prefix, local_metadata, strict,
        missing_keys, upexpected_keys, error_msgs):
    if prefix+"pe" in state_dict:
        model.register_buffer('pe', state_dict[prefix+"pe"])
    else:
        state_dict[prefix+"pe"] = model.pe

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding: dict, dropout: float, max_len:int, 
        factorize:bool=False):
        """
        Parameters
        ----------
        embedding: dict
            Input to nn.Embedding
        dropout: float
            Dropout after positional encoding
        factorize: bool
            True for old Transformer, False for normal Transformer.
        """
        super().__init__()
        self.embedding = nn.Embedding(**embedding)
        self.emb_size = embedding['embedding_dim']
        self.max_len = max_len
        self.factorize = factorize
        if self.factorize:
            self.factor = math.sqrt(self.emb_size)
        self.dropout = nn.Dropout(p=dropout)
        pe = get_posenc(max_len, self.emb_size)
        self.register_buffer('pe', pe)
        self._register_load_state_dict_pre_hook(load_pe_pre_hook_load, with_module=True)
    
    def forward(self, input: torch.Tensor, position: int=None):
        """
        Transpose is included here.

        Parameters
        ----------
        input: (torch.long)[batch_size, length]
        position(->None): int or None

        Returns
        -------
        output(torch.float)[length, batch_size, embedding_dim]: 
        """
        input = self.embedding(input.transpose(0, 1).contiguous())
        length = input.shape[0]
        if length > self.max_len:
            print("[WARNING] overflow in Positional embedding. PE is extended.")
            pe = get_posenc(length=length, emb_size=self.emb_size).to(self.pe.device)
            self.register_buffer('pe', pe)
            self.max_len = length
        if self.factorize:
            input *= self.factor
        if position is None:
            pe = Variable(self.pe[:input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input+pe)

# encoders
class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layer, norm=None, init=dict()):
        """
        AttentionEncoderと同じ。

        Parameters
        ----------
        layer: dict
            Parameters for SelfAttentionLayer
        n_layer: int
        norm: dict or None
            Parameters for nn.LayerNorm
        init: dict
            Initialization for each name in self.encoder.layers[i].state_dict()
        """
        super().__init__()
        d_model = layer['d_model']
        layer = SelfAttentionLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, norm=norm)

        # weight init
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    init_config2func(config)(param)
    
    def forward(self, src, key_padding_mask):
        """
        Parameters
        ----------
        src: (torch.float)[length, batch_size, d_model]
        key_padding_mask: (torch.float)[batch_size, length]

        Returns
        -------
        memory: (torch.float)[length, batch_size, d_model]
        """
        return self.encoder(src=src, mask=None, src_key_padding_mask=key_padding_mask)

class LatentSequenceDecoder(nn.Module):
    def forward(self, mode='forced', *args, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Mode to forward
        args, kwargs: 
            See each function for details.
        """
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            raise ValueError(f'Unsupported type of mode: {mode}')
    def forced(self, *args, **kwargs):
        raise NotImplementedError
    def cell_forward(self, *args, **kwargs):
        raise NotImplementedError
    def prepare_cell_forward(self, *args, **kwargs):
        raise NotImplementedError

class AttentionDecoder(LatentSequenceDecoder):
    def __init__(self, layer, num_layers, init, max_len, load_square_mask='keep'):
        """
        layer: dict
            input for SelfAttentionLayer
        num_layers: int
        init: dict
            Initialization for each parameter
        max_len: int
        load_square_mask: いる?
        """
        super().__init__()
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('square_subsequent_mask', square_mask)
        d_model = layer['d_model']
        self.d_model = d_model

        # decoder
        decoder_layer = SelfAttentionLayer(**layer)
        self.decoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)
        
        # weight init
        for layer in self.decoder.layers:
            for param_name in init:
                init_config2func(init[param_name])(layer.state_dict()[param_name]) 

        # define operation in load_state_dict
        if load_square_mask == 'keep':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'load':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict:
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'larger':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict and \
                    len(model.square_subsequent_mask) < len(state_dict[prefix+"square_subsequent_mask"]):
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        else:
            raise ValueError(f"Unsupported type of config.load_square_mask: {load_square_mask}")
        self._register_load_state_dict_pre_hook(pre_hook, with_module=True)
 
    def prepare_cell_forward(self, latent: torch.Tensor):
        """
        Parameters
        ----------
        latent: torch.tensor(float)[batch_size, d_model]

        Returns
        -------
        state: [torch.tensor(float)[0, batch_size, d_model]]
        
        """
        batch_size, d_model = latent.shape
        return [torch.zeros(size=(0, batch_size, self.d_model), dtype=torch.float, device=latent.device)
            for i_layer in range(self.decoder.num_layers)]
    def gather_beam(self, state, beam_index: torch.Tensor):
        """
        Parameters
        ----------
        state: list[(float)[length, batch_size*beam_size, d_model]]
        beam_index: (long)[batch_size, beam_size]
        """
        length, _, d_model = state[0].shape
        batch_size, beam_size = beam_index.shape
        new_state = []
        beam_index = beam_index.view(1, batch_size, beam_size, 1).expand(length, -1, -1, d_model)
        for state0 in state:
            state0 = state0.view(length, batch_size, beam_size, d_model)
            state0 = state0.gather(dim=2, index=beam_index)
            state0 = state0.view(length, -1, d_model)
            new_state.append(state0)
        return new_state

    # tgt, memory, memory_key_padding_mask

    def forced(self, tgt, latent):
        """
        Parameters
        ----------
        tgt: (float)[max_len, batch_size, d_model]
        latent (float)[batch_size, d_model]

        Returns
        -------
        output: (float)[batch_size, max_len, d_model]
        """
        max_len, _, _ = tgt.shape
        tgt = tgt + latent.unsqueeze(0)
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=tgt, mask=input_mask, src_key_padding_mask=None)
        return output.transpose(0, 1)

    def cell_forward(self, tgt, latent, state, position):
        """
        Parameters
        ----------
        tgt: (float)[1, batch_size, d_model]
            embedded input at (position) th place.
        latent: (float)[batch_size, d_model]
            latent representation

        Returns
        -------
        cur_output(float)[batch_size, 1, d_model]:
            Output of decoder
        state: [(float)[length, batch_size, d_model])]
        """

        cur_output = (tgt + latent.unsqueeze(0))
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(cur_y, y, y, attn_mask=None,
                        key_padding_mask=None, need_weights=False)[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        
        return cur_output.transpose(0, 1), state

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n_class = input.shape[-1]
        return super().forward(input=input.contiguous().view(-1, n_class), target=target.ravel())

class GreedyDecoder(nn.Module):
    def __init__(self, start_token, end_token):
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self._device_param = nn.Parameter(torch.zeros((0,)))
    def forward(self, *args, mode, **kwargs):
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported type of mode: {mode}")
    def init(self, batch_size):
        cur_input = torch.full((batch_size, 1), fill_value=self.start_token,
            dtype=torch.long, device=self._device_param.device)
        return cur_input, []
    def add(self, cur_proba, outs):
        cur_input = torch.argmax(cur_proba, dim=-1)
        outs.append(cur_input)
        return cur_input, outs
    def aggregate(self, outs):
        return torch.cat(outs, dim=1)
    
    def beam_init(self, latent: torch.Tensor, beam_size: int):
        """
        Parameters
        ----------
        latent: (float)[batch_size, latent_size]
        beam_size: int

        Returns
        -------
        latent: (float)[batch_size*beam_size, latent_size]
        cur_input: (float)[batch_size*beam_size, 1]
        is_ended: (bool)[batch_size, beam_size]
        outs: (long)[0(length), batch_size, beam_size]
        proba: (float)[batch_size, beam_size]
        
        """
        batch_size, latent_size = latent.shape
        device = latent.device
        latent = latent.view(batch_size, 1, latent_size).expand(-1, beam_size, -1).contiguous()
        latent = latent.view(batch_size*beam_size, latent_size)
        cur_input = torch.full((batch_size*beam_size, 1), fill_value=self.start_token,
            dtype=torch.long, device=device)
        is_ended = torch.full((batch_size, beam_size), fill_value=False,
            dtype=torch.bool, device=device)
        outs = torch.zeros((0, batch_size, beam_size), dtype=torch.long, device=device)
        proba = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        return latent, cur_input, outs, proba, is_ended
    def beam_add(self, cur_proba: torch.Tensor, proba: torch.Tensor, 
            outs: torch.Tensor, is_ended: torch.Tensor):
        """
        Parameters
        ----------
        cur_proba: (float)[batch_size*beam_size, 1, voc_size]
        proba: (float)[batch_size, beam_size]
            ※softmax前
        outs: (long)[length, batch_size, beam_size]
        is_ended: (bool)[batch_size, beam_size]
        
        B: batch_size
        E: beam_size
        V: voc_size
        """
        _, _, voc_size = cur_proba.shape
        length, batch_size, beam_size = outs.shape
        cur_proba = cur_proba.view(batch_size, beam_size, voc_size)

        # mask ended probas
        cur_proba[is_ended] = -torch.inf
        cur_proba[:,:,self.end_token][is_ended] = 0
        proba = proba.unsqueeze(-1) + cur_proba
        proba = proba.view(batch_size, -1) # [B, E*V]
        proba, topk_beam_voc = proba.topk(k=beam_size, dim=-1) # [B, E]
        topk_voc = topk_beam_voc % voc_size # [B, E]
        topk_beam = torch.div(topk_beam_voc, voc_size, rounding_mode='floor') # [B, E]

        # gather values
        outs = outs.gather(dim=-1, index=topk_beam.view(1, batch_size, beam_size) \
            .expand((length, batch_size, beam_size)))
        is_ended = is_ended.gather(dim=-1, index=topk_beam)

        outs = torch.cat([
            outs,
            topk_voc.unsqueeze(0)
        ], dim=0)
        is_ended[topk_voc == self.end_token] = True
        cur_input = topk_voc.view(batch_size*beam_size, 1)
        return cur_input, proba, outs, is_ended, topk_beam
    def beam_aggregate(self, outs: torch.Tensor):
        """
        Parameters
        ----------
        outs: (long)[length, batch_size, beam_size]

        Returns
        -------
        outs: (long)[batch_size, length]
        """
        return outs[:,:,0].transpose(0, 1).contiguous()

    def force(self, proba, add_start_token=False):
        force = torch.argmax(proba, dim=-1)
        if add_start_token:
            batch_size, length = force.shape
            force = torch.cat([torch.full((batch_size, 1), fill_value=self.start_token),
                force])
        return force
