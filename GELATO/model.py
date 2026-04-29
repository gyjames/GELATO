import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

def read_glove_vecs(pickle_file_path):
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"Pickle file {pickle_file_path} not found.")
    try:
        with open(pickle_file_path, 'rb') as f:
            word_to_vec_map = pickle.load(f)
        return word_to_vec_map
    except pickle.PickleError as e:
        raise pickle.PickleError(f"Error loading pickle file {pickle_file_path}: {str(e)}")

def load_pretrained_embeddings(word_to_vec_map, word_to_index, embedding_dim):
    vocab_size = len(word_to_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            embedding_matrix[idx] = word_to_vec_map[word]
    return torch.FloatTensor(embedding_matrix)

class GELATO(nn.Module):
    def __init__(self, word_to_vec_map, w2i, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.padding_idx = 0
        if word_to_vec_map:
            pretrained_embeddings = load_pretrained_embeddings(word_to_vec_map, w2i, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=self.padding_idx)
        else:
            self.embedding = nn.Embedding(len(w2i), embedding_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, dropout=0.5, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.fc_out = nn.Linear(n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _create_padding_mask(self, text):
        return (text == self.padding_idx)

    def forward(self, text):
        padding_mask = self._create_padding_mask(text)
        embedded = self.embedding(text)
        attn_output, _ = self.self_attention(embedded, embedded, embedded, key_padding_mask=padding_mask)
        attn_output = self.layer_norm1(attn_output + embedded)
        attn_output = attn_output.unsqueeze(1)
        conved = self.conv(attn_output).squeeze(3)
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        output = self.dropout(pooled)
        output = F.relu(output)
        output = self.fc_out(output)
        return output

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'embedding' not in name:
                nn.init.xavier_uniform_(p)

class GELATO_r(nn.Module):
    def __init__(self, word_to_vec_map, w2i, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.padding_idx = 0
        if word_to_vec_map:
            pretrained_embeddings = load_pretrained_embeddings(word_to_vec_map, w2i, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=self.padding_idx)
        else:
            self.embedding = nn.Embedding(len(w2i), embedding_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, dropout=0.3, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.fc = nn.Linear(n_filters, 50)
        self.fc_out = nn.Linear(50, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _create_padding_mask(self, text):
        return (text == self.padding_idx)

    def forward(self, text):
        padding_mask = self._create_padding_mask(text)
        embedded = self.embedding(text)
        attn_output, _ = self.self_attention(embedded, embedded, embedded, key_padding_mask=padding_mask)
        attn_output = self.layer_norm1(attn_output + embedded)
        attn_output = attn_output.unsqueeze(1)
        conved = self.conv(attn_output).squeeze(3)
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        output = F.relu(pooled)
        output = self.fc(output)
        output = F.relu(output)
        output = self.fc_out(output)
        return output

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'embedding' not in name:
                nn.init.xavier_uniform_(p)