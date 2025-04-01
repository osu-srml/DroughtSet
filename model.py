import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch import nn, Tensor
import torch.nn.init as init

from torch.autograd import Variable


def build_model(args,device):
    print('build model')
    model = ModelForecast_SPDrought(args,device, num_categories = 96, 
        num_layers = args.num_layers, 
        time_dim = args.time_dim,
        time_rep_dim = args.time_rep_dim,
        embedding_dim = args.embedding_dim,
        num_heads = args.inner_att_heads,
        dim_feedforward = args.dim_feedforward,
        mlp_input=args.mlp_input,
        mlp_hidden = args.mlp_hidden,
        mlp_dim = args.mlp_dim,
        dropout=args.dropout,
        num_tasks= args.num_task,
        mask_length  = args.input_window,
        ).to(device)
    return model
        
        
        
class ModelForecast_SPDrought(nn.Module):
    def __init__(self, args, device, num_categories, num_layers, embedding_dim,
                 time_dim=15,
                 time_rep_dim=64,
                 num_heads=4,
                 dim_feedforward=512,
                 mlp_input=14,
                 mlp_hidden=16,
                 mlp_dim=32,
                 dropout=0,
                 num_tasks=1,
                 mask_length=1196,
                 batch_first=False):
        super(ModelForecast_SPDrought, self).__init__()
        self.args = args
        self.device = device
        self.batch_first = batch_first
        self.embedding_dim = embedding_dim

        # Expand input dimensions
        self.expand_dims = ExpandDimsMLP(time_dim, time_rep_dim, dropout_prob=dropout)
        self.pos_encoder = PositionalEncoding(d_model=time_rep_dim, max_len=mask_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=time_rep_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embeddings = nn.Embedding(num_categories + 2, embedding_dim, padding_idx=num_categories + 1)
        self.embeddings.weight.data[num_categories + 1] = torch.zeros(embedding_dim)

        self.mlp_static = nn.Sequential(
            nn.Linear(mlp_input, mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_dim, bias=False),
        )

        combined_dim = time_rep_dim + embedding_dim + mlp_dim
        self.mlp_tgt = nn.Linear(time_dim, combined_dim, bias=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=combined_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_d_layers)

        self.mask_enc = None if args.no_encoder_mask else self.generate_mask(mask_length, mask_length).to(device)
        self.mask_dec = self.generate_mask(mask_length + args.unused_time, mask_length).to(device)

        self.agg = BatchDistanceAwareAttentionAggregator(mlp_input)

        self.regressors = nn.ModuleList([
            nn.Linear(combined_dim, 1 if args.use_decoder else self.args.unused_time) for _ in range(num_tasks)
        ])

        self.pos_decoder = PositionalEncoding(d_model=combined_dim, max_len=mask_length + args.unused_time)

    def generate_mask(self, sz, window_end=80):
        indices = torch.arange(sz).unsqueeze(0) - torch.arange(sz).unsqueeze(1)
        return torch.where((indices >= 0) & (indices <= window_end), 0.0, float('-inf'))

    def forward(self, x_num_time, x_num_static, x_cata, tgt=None, task_idx=None, target_idx=None):
        if self.batch_first:
            x_num_time = x_num_time.permute(2, 0, 1)
            if self.args.use_decoder:
                tgt = tgt.permute(2, 0, 1)

        x_num_time_predictors = self.expand_dims(x_num_time)
        x_position_embedded = self.pos_encoder(x_num_time_predictors)

        x_time_representation = self.transformer_encoder(x_position_embedded, self.mask_enc)

        x_static_representation = self.mlp_static(x_num_static)
        x_static_representation = x_static_representation.unsqueeze(0).repeat(x_time_representation.size(0), 1, 1)

        x_cata[x_cata == -1] = self.embeddings.padding_idx
        x_cata_embedded = self.embeddings(x_cata).squeeze()
        x_cata_embedded = x_cata_embedded.unsqueeze(0).repeat(x_time_representation.size(0), 1, 1)

        combined = torch.cat([x_time_representation, x_cata_embedded, x_static_representation], dim=2)

        if self.args.use_decoder:

            pe_tgt = self.pos_decoder(self.mlp_tgt(tgt))

            decoded = self.transformer_decoder(pe_tgt, combined, tgt_mask=self.mask_dec)[-self.args.unused_time:].permute(1, 0, 2)

            if task_idx is not None:
                return self.regressors[task_idx](decoded)[:, target_idx:target_idx + 1]
            else:
                return [reg(decoded) for reg in self.regressors]

        decoded = combined[-1]
        return [reg(decoded) for reg in self.regressors]        
        
        
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class ExpandDimsMLP(nn.Module):
    def __init__(self, time_dim, time_rep_dim, dropout_prob=0):
        super(ExpandDimsMLP, self).__init__()
        self.linear = nn.Linear(time_dim, time_rep_dim)
        init.xavier_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class BatchDistanceAwareAttentionAggregator(nn.Module):
    def __init__(self, static_dim):
        super(BatchDistanceAwareAttentionAggregator, self).__init__()
        self.query = nn.Linear(static_dim, static_dim)
        self.key = nn.Linear(static_dim, static_dim)
        self.softmax = nn.Softmax(dim=1)  
        self.static_dim = static_dim
        
    def forward(self, current_static_feature, static_features, current_time_feature, time_features, distances, show=False):
        batch_size, num_positions, _ = static_features.size()

        queries = self.query(current_static_feature.view(batch_size, 1, -1))
        keys = self.key(static_features)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.static_dim ** 0.5)

        distance_factors = 1 / (distances + 1e-9)
        adjusted_attention_scores = attention_scores * distance_factors.unsqueeze(1)

        attention_weights = torch.softmax(adjusted_attention_scores, dim=2)

        neighbor_weights = torch.cat((attention_weights[:, :, :12], attention_weights[:, :, 13:]), dim=2)
        neighbor_time_features = torch.cat((time_features[:, :12], time_features[:, 13:]), dim=1)

        weighted_time_features = torch.einsum('bij,bjkl->bikl', neighbor_weights, neighbor_time_features).squeeze()

        current_feature_weighted = current_time_feature * attention_weights[:, 0, 12].unsqueeze(-1).unsqueeze(-1)

        aggregated_feature = current_feature_weighted + weighted_time_features

        return (aggregated_feature, attention_weights) if show else aggregated_feature
