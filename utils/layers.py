import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


class CVGCNLayer(nn.Module):

    def __init__(self,
                 num_nodes,
                 num_class,
                 input_dim,
                 labels,
                 dropout,
                 device,
                 train_mask,
                 sr_weight=0,
                 extra_dim=False,
                 select='soft',
                 reciprocal=False,
                 layer_id=0):
        super(CVGCNLayer, self).__init__()

        self.register_buffer("mag", torch.randn(1, input_dim))
        self.layer_id = layer_id
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.labels = labels.to(device)
        self.dropout = dropout
        self.device = device
        self.train_mask = train_mask
        self.MLP = ComplexLinear(input_dim, input_dim)

    def forward(self, x, adj):
        features = x
        adj = torch.complex(adj, torch.zeros(adj.shape).to(self.device))
        message = torch.matmul(adj, features)
        new_fea = self.MLP(message) + features
        separate_loss = torch.tensor(0)
        supervised_loss = torch.tensor(0)
        return new_fea, separate_loss, supervised_loss


class AngularAggLayer(nn.Module):

    def __init__(self,
                 num_nodes,
                 num_class,
                 input_dim,
                 labels,
                 dropout,
                 device,
                 train_mask,
                 sr_weight=0,
                 extra_dim=False,
                 select='soft',
                 reciprocal=False,
                 layer_id=0):
        super(AngularAggLayer, self).__init__()

        self.register_buffer("mag", torch.randn(1, input_dim))
        self.layer_id = layer_id
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.labels = labels.to(device)
        self.sr_weight = sr_weight
        self.dropout = dropout
        self.reciprocal = reciprocal
        self.select_mode = select
        self.num = num_class + extra_dim

        self.theta = torch.nn.Parameter(
            torch.complex(torch.ones(self.num), torch.zeros(self.num)))

        self.device = device
        self.train_mask = train_mask
        self.MLP = ComplexLinear(input_dim, input_dim)
        self.to_class = ComplexLinear(input_dim, self.num)

        self.select = select

        one_hot_labels = torch.nn.functional.one_hot(
            self.labels, num_classes=self.num_class).to(self.device)
        self.one_hot_labels = one_hot_labels
        index_mask = torch.ones(one_hot_labels.shape[0])
        index_mask[self.train_mask] = 0
        index_mask = index_mask.unsqueeze(1).repeat(1, self.num_class).to(
            self.device).bool()
        self.index_mask = index_mask

    def build_type_adj(self, weight, label):
        label_indices = label.long()
        A_hat = torch.index_select(weight, 0, label_indices).to(self.device)
        A_hat = torch.index_select(A_hat, 1, label).to(self.device)
        return A_hat

    def build_similarity_matrix(self, feature, adj=None):
        pseudo_norm = torch.abs(self.to_class(feature))
        pseudo_labels = F.softmax(pseudo_norm, dim=1)
        similarity_matrix = pseudo_labels

        sorted_theta, indices = torch.sort(self.theta.imag)
        difference = torch.abs(sorted_theta[:-1] - sorted_theta[1:])

        max_value, _ = torch.max(difference, dim=0)
        difference = torch.sum(difference) / (max_value + 1e-6)
        separate_loss = self.sr_weight / (1e-6 + difference.sum())

        supervised_loss = F.cross_entropy(pseudo_norm[self.train_mask],
                                          self.labels[self.train_mask]) * 0.1

        return similarity_matrix, separate_loss, supervised_loss

    def att(self, similarity_matrix):

        similarity_matrix = torch.complex(
            similarity_matrix,
            torch.zeros(similarity_matrix.shape).to(self.device))

        in_theta = self.theta

        if self.reciprocal:
            out_theta = in_theta.clone()
        else:
            out_theta = in_theta.conj()

        if self.select_mode == 'soft':
            in_type = torch.matmul(similarity_matrix, in_theta)
            out_type = torch.matmul(similarity_matrix, out_theta)
        elif self.select_mode == 'hard':
            norm_matrix = torch.abs(similarity_matrix)
            arg_max = torch.argmax(norm_matrix, dim=1)
            in_type = in_theta[arg_max]
            out_type = out_theta[arg_max]

        M = torch.matmul(in_type.unsqueeze(1), out_type.unsqueeze(0))
        return M, in_type, out_type

    def forward(self, x, adj, save_final_adj=False):
        features = x

        similarity_matrix, separate_loss, supervised_loss = self.build_similarity_matrix(
            features, adj)
        A_hat, in_type, out_type = self.att(similarity_matrix)
        adj = torch.mul(adj, A_hat)
        message = torch.matmul(adj, features)
        new_fea = self.MLP(message) + features

        return new_fea, separate_loss, supervised_loss


class ComplexLinear(nn.Module):
    """
    $$y = (ac - bd) + (ad + bc)i$$
    where $a$ and $b$ are the real and imaginary parts of the input $x$, and $c$ and $d$ are the weights of the `self.fc_r` and `self.fc_i` layers, respectively.
    """

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

        init.uniform_(self.fc_i.weight, a=-torch.pi / 4, b=torch.pi / 4)

    def forward(self, input):
        dtype = torch.complex64
        return (self.fc_r(input.real) - self.fc_i(input.imag)).type(dtype) \
            + 1j * (self.fc_r(input.imag) + self.fc_i(input.real)).type(dtype)


class ComplexDropout(nn.Module):

    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def complex_dropout(self, input, p=0.5, training=True):
        device = input.device

        real = input.real
        imag = input.imag
        real = F.dropout(real, p, training)
        imag = F.dropout(imag, p, training)
        return torch.complex(real, imag)

    def forward(self, input):
        if self.training:
            return self.complex_dropout(input, self.p)
        else:
            return input


def all_relu(x):
    return F.relu(x.real).type(
        torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)


def img_relu(x):
    return x.real.type(
        torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)


def real_relu(x):
    return F.relu(x.real).type(
        torch.complex64) + 1j * x.imag.type(torch.complex64)
