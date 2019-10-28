import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layer=1):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=n_layer)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class SelfAttention(torch.nn.Module):
    def __init__(self, qkv_dimensions, hidden_size=256, n_heads=4, output_dim=None, dropout=0.1, normaliza_qk=False):
        super(SelfAttention, self).__init__()
        print('Enter Self Attention')
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of heads.")
        self.dropout = dropout
        self.dropout = nn.Dropout(dropout) if 0 < dropout < 1 else None
        self.normalize_qk = normaliza_qk

        q_dim, k_dim, v_dim = qkv_dimensions
        self.q_proj = nn.Linear(q_dim, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(k_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(v_dim, self.hidden_size, bias=False)

        if output_dim is None:
            self.output_transform = None
        else:
            self.output_transform = nn.Linear(self.hidden_size, output_dim, bias=False)

    @property
    def depth(self):
        return self.hidden_size // self.n_heads

    def forward(self, q, k, v):
        k_equal_q = k is None
        if self.q_proj is not None:
            q = self.q_proj(q)
        if k_equal_q:
            k = q
        elif self.k_proj is not None:
            k = self.k_proj(k)
        if self.v_proj is not None:
            v = self.v_proj(v)
        if self.n_heads > 1:
            q = self._split_heads(q)
            if not k_equal_q:
                k = self._split_heads(k)
            v = self._split_heads(v)
        if self.normalize_qk:
            q = q / torch.norm(q, dim=-1).unsqueeze(-1)
            if not k_equal_q:
                k = k / torch.norm(k, dim=-1).unsqueeze(-1)
        if k_equal_q:
            k = q
        q = q * self.depth ** -0.5

        # q, k, v  : [num_heads x B, T, depth]
        logits = torch.bmm(q, k.transpose(1, 2))
        weights = F.softmax(logits, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        attention_output = torch.bmm(weights, v)
        attention_output = self._combine_heads(attention_output)
        if self.output_transform is not None:
            attention_output = self.output_transform(attention_output)
        return attention_output

    def _split_heads(self, x):
        time_step = x.shape[1]
        return (
            x.view(-1, time_step, self.n_heads, self.depth)
                .transpose(1, 2).contiguous()
                .view(-1, time_step, self.depth)
        )

    def _combine_heads(self, x):
        time_step = x.shape[1]
        return (
            x.view(-1, self.n_heads, time_step, self.depth)
                .transpose(1, 2).contiguous()
                .view(-1, time_step, self.hidden_size)
        )


class Encoder_with_SelfAttn(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layer=2):
        assert n_layer > 1
        print('Enter Encoder with Self Attention')
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=n_layer)

        qkv_dimensions = [enc_hid_dim, enc_hid_dim, enc_hid_dim + emb_dim]
        self.self_attentions = torch.nn.ModuleList([
            SelfAttention(qkv_dimensions, enc_hid_dim, n_heads=4)
            for _ in range(n_layer - 1)
        ])

        input_dimensions = [emb_dim] + [enc_hid_dim] * (n_layer - 1)
        self.rnns = torch.nn.ModuleList([nn.GRU(
            dim, enc_hid_dim, 1,
            batch_first=True, bidirectional=True  # batch first 的问题需要改一下
        ) for dim in input_dimensions])

        self.bidirectional_projections = torch.nn.ModuleList(
            [nn.Linear(enc_hid_dim * 2, enc_hid_dim, bias=False)
             for _ in range(n_layer)])

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]
        src = src.transpose(0, 1)

        embedded = self.embedding_dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        net = embedded
        for i, rnn in enumerate(self.rnns):
            net, final_state = rnn(net, None)
            if self.bidirectional_projections is not None and i < len(self.rnns) - 1:
                net = self.bidirectional_projections[i](net)
            if self.self_attentions is not None and i < len(self.rnns) - 1:
                net = self.self_attentions[i](net, net, torch.cat([embedded, net], dim=2))
        # net = [bs, len, hid_dim * 2]
        outputs = net.transpose(0, 1)

        # outputs = [src sent len, batch size, hid dim * num directions]

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(net[:, -1, :]))

        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    MAX_DECODE_LEN = 100

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=1.0):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        if trg is not None:
            max_len = trg.shape[0]
        else:
            assert teacher_forcing_ratio == 0
            max_len = Seq2Seq.MAX_DECODE_LEN

        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = src[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    # model config
    INPUT_DIM = 1000
    OUTPUT_DIM = 1000
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    # enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, n_layer=2)
    # dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    # model = Seq2Seq(enc, dec, device).to(device)
    #
    # # src = [src sent len, batch size]
    # # trg = [trg sent len, batch size]
    # src = torch.randint(0, 1000, (30, 16)).cuda()
    # trg = torch.randint(0, 1000, (30, 16)).cuda()
    #
    # model(src, trg)
