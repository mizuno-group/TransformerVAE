from  .models import *

import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

from .modules.tunnel import *
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

from .modules.sequence import *
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, AttentionDecoder, GreedyDecoder, CrossEntropyLoss]:
    module_type2class[cls.__name__] = cls

from .modules.vae import *
for cls in [VAE, MinusD_KLLoss]:
    module_type2class[cls.__name__] = cls

from .modules.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler]:
    module_type2class[cls.__name__] = cls