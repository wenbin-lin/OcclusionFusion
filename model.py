import torch
import torch.nn.functional as F
from torch_geometric.nn import DeepGCNLayer, TransformerConv
from torch.nn import Linear, LayerNorm, ReLU, LSTM


class MotionCompleteNet(torch.nn.Module):
    def __init__(self):
        super(MotionCompleteNet, self).__init__()
        feature_dim = 11
        hidden_channels = 32
        output_dim = 4
        self.hidden_channels = hidden_channels

        self.node_encoder = Linear(feature_dim, hidden_channels)

        self.lstm_layer_num = 2
        self.lstm_hidden_dim = 32
        self.lstm_output_dim = 4
        self.seq_encoder = LSTM(input_size=4, hidden_size=self.lstm_hidden_dim, num_layers=self.lstm_layer_num, batch_first=False)
        self.seq_linear = Linear(self.lstm_hidden_dim, self.lstm_output_dim)

        self.conv0 = TransformerConv(hidden_channels, hidden_channels)

        self.layer11 = self.build_layer(hidden_channels)
        self.layer12 = self.build_layer(hidden_channels)
        self.layer21 = self.build_layer(hidden_channels)
        self.layer22 = self.build_layer(hidden_channels)
        self.layer31 = self.build_layer(hidden_channels)
        self.layer32 = self.build_layer(hidden_channels)
        self.layer41 = self.build_layer(hidden_channels)
        self.layer42 = self.build_layer(hidden_channels)
        self.layer51 = self.build_layer(hidden_channels * 2)
        self.layer52 = self.build_layer(hidden_channels * 2)
        self.layer61 = self.build_layer(hidden_channels * 3)
        self.layer62 = self.build_layer(hidden_channels * 3)
        self.layer71 = self.build_layer(hidden_channels * 4)
        self.layer72 = self.build_layer(hidden_channels * 4)

        self.norm_out = LayerNorm(hidden_channels * 4, elementwise_affine=True)
        self.act_out = ReLU(inplace=True)

        self.lin = Linear(hidden_channels * 4, output_dim)

    def build_layer(self, ch):
        conv = TransformerConv(ch, ch)
        norm = LayerNorm(ch, elementwise_affine=True)
        act = ReLU(inplace=True)
        layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=False)
        return layer

    def forward(self, curr_pos, curr_motion, prev_motion, edge_indexes, down_sample_maps, up_sample_maps):
        node_num = curr_pos.shape[0]

        seq_feature, _ = self.seq_encoder(prev_motion.view(-1, node_num, 4), None)

        seq_pred = self.seq_linear(seq_feature[-1]).view(-1, self.lstm_output_dim)

        # the input feature of nodes
        x = self.node_encoder(torch.cat([curr_pos, seq_pred, curr_motion], dim=-1))

        feature0 = self.conv0(x, edge_indexes[0])
        feature1 = self.layer11(feature0, edge_indexes[0])
        feature1 = self.layer12(feature1, edge_indexes[0])

        feature2 = feature1[down_sample_maps[0]]
        feature2 = self.layer21(feature2, edge_indexes[1])
        feature2 = self.layer22(feature2, edge_indexes[1])

        feature3 = feature2[down_sample_maps[1]]
        feature3 = self.layer31(feature3, edge_indexes[2])
        feature3 = self.layer32(feature3, edge_indexes[2])

        feature4 = feature3[down_sample_maps[2]]
        feature4 = self.layer41(feature4, edge_indexes[3])
        feature4 = self.layer42(feature4, edge_indexes[3])

        feature5 = feature4[up_sample_maps[2]]
        feature5 = self.layer51(torch.cat([feature5, feature3], dim=-1), edge_indexes[2])
        feature5 = self.layer52(feature5, edge_indexes[2])

        feature6 = feature5[up_sample_maps[1]]
        feature6 = self.layer61(torch.cat([feature6, feature2], dim=-1), edge_indexes[1])
        feature6 = self.layer62(feature6, edge_indexes[1])

        feature7 = feature6[up_sample_maps[0]]
        feature7 = self.layer71(torch.cat([feature7, feature1], dim=-1), edge_indexes[0])
        feature7 = self.layer72(feature7, edge_indexes[0])

        out = self.act_out(self.norm_out(feature7))
        out = F.dropout(out, p=0.1, training=self.training)

        pred = self.lin(out)

        # use softplus to make sigma positive
        pred[:, -1] = F.softplus(pred[:, -1])

        return pred
