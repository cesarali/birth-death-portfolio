import torch
import numpy as np
from torch import nn


class EncodeOntoRWRNN(EncoderRNN):
    def __init__(self, vocab, fix_len, latent_dim, n_symbols, n_symbol_classes, **kwargs):
        super(EncodeOntoRWRNN, self).__init__(vocab, fix_len, n_symbols, input_dim=0, **kwargs)

        self.rw_length = kwargs.get("rw_length")
        # Symbols
        self.symbols = nn.Parameter(torch.Tensor(n_symbols, latent_dim))
        torch.nn.init.normal_(self.symbols)

        self.n_symbol_classes = n_symbol_classes
        if self.n_symbol_classes > 1:
            self.get_symbols_class_logits = nn.Linear(self.last_dim_before_output, n_symbol_classes)

        self.get_rw_starting_point = nn.Linear(self.last_dim_before_output, n_symbols)
        self.graph_generator = create_instance('graph_generator', kwargs, *(latent_dim, n_symbols))

    def forward_from_graph(self,input,adj_matrix,tau):
        # embed input
        x, l = input
        x = self.embedding(x)  # [B, T, D]
        t_len = x.size(1)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, True, False)

        # RNN (bi-directional LSTM):
        out, self.hidden_state = self.rnn_cell(x, self.hidden_state)  # [B, T, 2*D], [2, 2, B, D]

        if self.attention:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, True, total_length=t_len)
            h, _ = self.attention_layer(out)  # [B, 2*D]
        else:
            h = self.hidden_state[0]  # [2, B, D]
            h = torch.cat((h[0], h[1]), dim=1)  # Merge h for both directions [B, 2*D]
            h = self.highway_network(h)

        # walks starting points:
        f0 = nn.functional.softmax(self.get_rw_starting_point(h), dim=-1)
        z = gumbel_softmax(f0, tau, self.device)  # [B, n_symbols]
        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        # get sentence-dependent node attributes
        f = torch.sigmoid(self.output(h))  # [B, n_symbols]

        # uniform probability over nodes:
        n_symbols = f.shape[1]
        pi = torch.full(f.shape, 1.0 / n_symbols, device=self.device)  # [B, n_symbols]

        # TODO: implement a way to get symbols classes
        c = None

        # random walks:
        # with torch.autograd.detect_anomaly():
        for i in range(1, self.rw_length):
            transition_prob = torch.matmul(z, adj_matrix) * f  # [B, n_symbols]

            # (*) condition depending on neighbor existence:
            torch_sum = torch.sum(transition_prob, dim=-1)
            cond = (torch_sum > 0.0).view(-1, 1).float()
            norm = torch_sum.view(-1, 1) + (1.0 - cond)
            transition_prob = cond * transition_prob / norm + (1 - cond) * pi

            # (*) sample step
            z = gumbel_softmax(transition_prob, tau, self.device)
            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

        return walks, c  # [B, L, n_symbols]

    def forward(self, input, tau, epsilon=1e-10):
        """
        input: input data (x): [B, T]
               temperature Gumbel distribution (tau): int
        Returns: one-hot sequence over symbols (z): [B, L, n_symbols]
        Notation:
        B: batch size; T: seq len (== fix_len); L: random walk length; Z: latent_dim
        """
        # sampling posterior matrix
        logits_matrix = self.graph_generator(self.symbols)
        adj_matrix = self.graph_generator.sample(logits_matrix, tau)
        return self.forward_from_graph(input,adj_matrix,tau)


class EncodeOntoRWCNN(EncoderCNN):
    def __init__(self, vocab, fix_len, latent_dim, n_symbols, n_symbol_classes, **kwargs):
        super(EncodeOntoRWCNN, self).__init__(vocab, fix_len, n_symbols, **kwargs)

        self.rw_length = kwargs.get("rw_length")
        out_channels = kwargs.get('output_channels')

        # Symbols
        self.symbols = nn.Parameter(torch.Tensor(n_symbols, latent_dim))
        torch.nn.init.normal_(self.symbols)

        self.n_symbol_classes = n_symbol_classes
        if self.n_symbol_classes > 1:
            self.get_symbols_class_logits = nn.Linear(out_channels[-1], n_symbol_classes)

        self.get_rw_starting_point = nn.Linear(self._hidden_dim, n_symbols)
        self.graph_generator = create_instance('graph_generator', kwargs, *(latent_dim, n_symbols))

    def forward_from_graph(self,input,adj_matrix,tau):
        # embed input
        x, _ = input
        x = self.embedding(x)  # [B, T, D]
        x = torch.unsqueeze(x, 1)

        # convolution
        y = self.layers(x)  # [B, D', L, 1]
        y = torch.squeeze(y, 3)
        y = self.nonlinearity(y)

        if self.bottleneck:
            if self.attention:
                y = y.permute(0, 2, 1)
                y, _ = self.attention_layer(y)  # [B, D']
            else:
                y = y.view(-1, self.flat_dim)  # [B, D'*L]
        else:
            print("This encoder must have bottleneck set to True")
            raise Exception

        # walks starting points:
        f0 = nn.functional.softmax(self.get_rw_starting_point(y))
        z = gumbel_softmax(f0, tau, self.device)  # [B, n_symbols]
        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        # get sentence-dependent node attributes
        f = torch.sigmoid(self.output_layer(y))  # [B, n_symbols]

        # uniform probability over nodes:
        n_symbols = f.shape[1]
        pi = torch.full(f.shape, 1.0 / n_symbols, device=self.device)  # [B, n_symbols]

        #TODO: implement a way to get symbols classes
        c = None

        # random walks:
        # with torch.autograd.detect_anomaly():
        for i in range(1, self.rw_length):
            transition_prob = torch.matmul(z, adj_matrix) * f  # [B, n_symbols]

            # (*) condition depending on neighbor existence:
            torch_sum = torch.sum(transition_prob, dim=-1)
            cond = (torch_sum > 0.0).view(-1, 1).float()
            norm = torch_sum.view(-1, 1) + (1.0 - cond)
            transition_prob = cond * transition_prob / norm + (1 - cond) * pi

            # (*) sample step
            z = gumbel_softmax(transition_prob, tau, self.device)
            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

        return walks, c  # [B, L, n_symbols]

    def forward(self, input, tau, epsilon=1e-10):
        """
        input: input data (x): [B, T]
               temperature Gumbel distribution (tau): int
        Returns: one-hot sequence over symbols (z): [B, L, n_symbols]
        Notation:
        B: batch size; T: seq len (== fix_len); L: random walk length; Z: latent_dim
        """
        # sampling posterior matrix
        logits_matrix = self.graph_generator(self.symbols)
        adj_matrix = self.graph_generator.sample(logits_matrix, tau)
        return self.forward_from_graph(input,adj_matrix,tau)

