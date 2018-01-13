import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, dim, emb_dim, num_emb, activation = nn.ReLU(),
            group_size=2, kernel_size=3, stride=1):
        super(Encoder, self).__init__()


        self.embeddings = nn.Embeddings(num_emb, emb_dim)
        self.activation = activation
        self.dim = dim

        padding = (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)

        self.prefix_group = nn.ModuleList([
            nn.Conv1d(emb_dim, emb_dim, kernel_size, stride, padding)
            for _ in range(group_size)])

        self.recursion_group = nn.ModuleList([
            nn.Conv1d(emb_dim, emb_dim, kernel_size, stride, padding)
            for _ in range(group_size)])

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        self.postfix_group = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(group_size)])

    def forward(self, data):

        hid = self.embeddings(data)
        for op in self.prefix_group:
            hid = self.activation(op(hid))

        while hid.size()[-1] > 4:
            for op in self.recursion_group:
                hid = self.activation(op(hid))
            hid = self.max_pool(hid)

        for op in self.postfix_group:
            hid = self.activation(op(hid))

        return hid.view([hid.size(0), self.dim])


class Decoder(nn.Module):
    def __init__(self, input_dim, num_emb, feature_size,
            activation = nn.ReLU(), group_size=2, kernel_size=3, stride=1):
        super(Decoder, self).__init__()

        self.activation = activation
        self.dim = input_dim

        padding = (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)

        self.prefix_group = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(group_size)])

        self.expand_conv = nn.Conv1d(feature_size, 2 * feature_size, kernel_size, stride, padding)
        self.recursion_group = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size, stride, padding)
            for _ in range(group_size - 1)])

        self.postfix_group = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size, stride, padding)
            for _ in range(group_size)])

        self.projection = nn.Linear(feature_size, num_emb)

    def forward(self, data, rec_steps):

        hid = data.view([hid.size(0), feature_size, -1])
        for op in self.prefix_group:
            hid = self.activation(op(hid))

        for _ in range(rec_steps):
            hid = self.activation(self.expand_conv(hid))
            hid = hid.view([hid.size(0), hid.size(1) // 2, -1])

            for op in self.recursion_group:
                hid = self.activation(op(hid))

        for op in self.postfix_group:
            hid = self.activation(op(hid))

        hid = self.projection(hid.transpose(1, 2))

        return hid


