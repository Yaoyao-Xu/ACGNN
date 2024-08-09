import torch
import torch.nn as nn
from utils.layers import AngularAggLayer, ComplexDropout, ComplexLinear, all_relu, img_relu, real_relu


class CGNN(nn.Module):

    def __init__(self,
                 num_features,
                 num_layers,
                 num_hidden,
                 num_class,
                 dropout,
                 labels,
                 num_nodes,
                 device,
                 train_mask,
                 sr_weight=0.5,
                 extra_dim=0,
                 select: str = 'soft',
                 activation='all',
                 reciprocal=False):
        super(CGNN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                AngularAggLayer(num_nodes,
                                num_class,
                                num_hidden,
                                labels,
                                dropout,
                                device,
                                train_mask,
                                sr_weight=sr_weight,
                                extra_dim=extra_dim,
                                select=select,
                                reciprocal=reciprocal,
                                layer_id=i))
        self.fcs = nn.ModuleList()

        self.fcs.append(ComplexLinear(num_features, num_hidden))

        self.fcs.append(ComplexLinear(num_hidden, num_class))

        self.labels = labels
        self.dropout = ComplexDropout(dropout)

        if activation == 'all':
            self.activation = all_relu
        elif activation == 'img':
            self.activation = img_relu
        elif activation == 'real':
            self.activation = real_relu

    def forward(self, x, adj):
        _layers = []
        hidden = self.activation(self.fcs[0](x))
        _layers.append(hidden)
        all_separate_loss = 0
        all_supervised_loss = 0
        for i, con in enumerate(self.convs):
            hidden, separate_loss, supervised_loss = con(hidden, adj)
            hidden = self.dropout(hidden)
            hidden = self.activation(hidden)

            all_separate_loss += separate_loss
            all_supervised_loss += supervised_loss

        hidden_angle = torch.angle(hidden)
        hidden_norm = torch.abs(hidden)

        output = self.fcs[-1](hidden)

        return hidden, output, hidden_angle, hidden_norm, all_separate_loss, all_supervised_loss
