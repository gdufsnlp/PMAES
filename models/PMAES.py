import torch
import torch.nn as nn
from torch.nn import functional as F


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))

        weight = self.v(w)
        weight = weight.squeeze(dim=-1)

        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(dim=-1)
        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))

        out = torch.sum(out, dim=1)

        return out


class EssayEncoder(nn.Module):
    def __init__(self, args, max_num, max_len, embed_dim, pos_vocab=None):
        super(EssayEncoder, self).__init__()
        self.N = max_num
        self.L = max_len
        self.args = args
        self.embed_dim = embed_dim

        self.embed_layer = nn.Embedding(num_embeddings=len(pos_vocab), embedding_dim=embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=args.filter_num, kernel_size=args.kernel_size)
        self.lstm = nn.LSTM(input_size=args.filter_num, hidden_size=args.lstm_units, num_layers=1, batch_first=True)
        self.word_att = SoftAttention(args.filter_num)
        self.sent_att = SoftAttention(args.lstm_units)

    def forward(self, x):
        embed = self.embed_layer(x)
        embed = nn.Dropout(self.args.dropout)(embed)
        embed = embed.view(embed.size()[0], self.N, self.L, self.embed_dim)
        sentence_fea = torch.tensor([], requires_grad=True).to(self.args.device)
        for n in range(self.N):
            sentence_embed = embed[:, n, :, :]
            sentence_cnn = self.conv1d(sentence_embed.permute(0, 2, 1))
            sentence_att = self.word_att(sentence_cnn.permute(0, 2, 1))
            sentence_att = nn.Dropout(self.args.dropout)(sentence_att)
            sentence_fea = torch.cat([sentence_fea, sentence_att.unsqueeze(1)], dim=1)

        essay_lstm, _ = self.lstm(sentence_fea)
        essay_fea = self.sent_att(essay_lstm)

        return essay_fea


class Scorer(nn.Module):
    def __init__(self, args):
        super(Scorer, self).__init__()
        self.fc_layer = nn.Linear(136, 50, bias=True)
        self.score_layer = nn.Linear(50, 1, bias=True)

    def forward(self, x):
        out = torch.tanh(self.fc_layer(x))
        out = torch.sigmoid(self.score_layer(out))

        return out


class PromptMappingCL(nn.Module):
    def __init__(self, args, tr_s_num, tr_t_num):
        super(PromptMappingCL, self).__init__()
        self.args = args
        self.temperate = 0.1
        self.source_project_head = nn.Linear(tr_t_num, args.lstm_units, bias=True)
        self.target_project_head = nn.Linear(tr_s_num, args.lstm_units, bias=True)

    def unsup_loss(self, x, size):
        label = torch.tensor([i for i in range(size, 2 * size)] + [i for i in range(0, size)], device=self.args.device)
        sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(x.shape[0], device=self.args.device)
        sim /= self.temperate
        loss = F.cross_entropy(sim, label)
        return loss

    def forward(self, source, target, s_essay_fea, t_essay_fea):
        s_fea = s_essay_fea.permute(1, 0)
        t_fea = t_essay_fea.permute(1, 0)
        s_anchor = source
        s_pos = source.mm(t_fea)
        s_pos = self.source_project_head(s_pos)
        s_cat = torch.cat([s_anchor, s_pos], dim=0)
        s_size = source.size()[0]
        s_loss = self.unsup_loss(s_cat, s_size)

        t_anchor = target
        t_pos = target.mm(s_fea)
        t_pos = self.target_project_head(t_pos)
        t_cat = torch.cat([t_anchor, t_pos], dim=0)
        t_size = target.size()[0]
        t_loss = self.unsup_loss(t_cat, t_size)

        loss = s_loss + t_loss

        return loss