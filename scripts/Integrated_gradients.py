import json
import torch
import re
import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence 
import argparse

import os
import json
from collections import defaultdict

def read_glove_vecs(file_path):  
    word_to_vec_map = {}  
    with open(file_path, 'r', encoding='utf-8') as f:  
        for line in f:  
            line = line.strip().split()  
            word = line[0]  
            vec = np.asarray(line[1:], dtype='float32')  
            word_to_vec_map[word] = vec  
    return word_to_vec_map  

def load_pretrained_embeddings(word_to_vec_map, word_to_index, embedding_dim):  
    vocab_size = len(word_to_index)  
    embedding_matrix = np.zeros((vocab_size, embedding_dim))  
    for word, idx in word_to_index.items():  
        if word in word_to_vec_map: 
            embedding_matrix[idx] = word_to_vec_map[word]  
    return torch.FloatTensor(embedding_matrix)

def tokenization(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '/', ':', ';', '<', '=', '>','\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    ls = [i.strip() for i in text.split()]
    for i, w in enumerate(ls):
        w = re.sub(r'\.$', '', w)
        ls[i] = w
    return ls

class GELATO(nn.Module):
    def __init__(self, embedding_pretrained, w2i, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        self.padding_idx = 0
        super().__init__()
        if embedding_pretrained:
            glove_path = f'vectors_dim{embedding_dim}.txt'
            word_to_vec_map = read_glove_vecs(glove_path)
            pretrained_embeddings = load_pretrained_embeddings(word_to_vec_map, w2i, embedding_dim)  
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=self.padding_idx)
        else:
            self.embedding = nn.Embedding(len(w2i), embedding_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, dropout=0.5, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (filter_sizes[0], embedding_dim))
        
        self.fc_out = nn.Linear(n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _create_padding_mask(self, text):
        mask = (text == self.padding_idx)
        return mask
    
    def forward(self, text):
        padding_mask = self._create_padding_mask(text)
        embedded = self.embedding(text)   
        attn_output, _ = self.self_attention(embedded, embedded, embedded,
                                            key_padding_mask=padding_mask)
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

class CustomSequenceDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return sentence, label 
    
def get_dataloader(sentences_path, labels, vocab, BATCH_SIZE, shuffle=False):
    sentences = []
    _labels = []
    data = zip(sentences_path, labels)
    for path, l in data:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        genome = tokenization(content)
        genome = [i for i in genome if i !='Cluster']
        sentence = []
        for gene in genome:
            try:
                sentence.append(vocab[gene])
            except:
                continue
        if len(sentence) > 100:
            sentences.append(sentence)
            _labels.append(l)
    labels = np.array(_labels)

    def collate_fn(batch):
        sentences = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        labels = [item[1] for item in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=vocab['<PAD>'])
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        return padded_sentences, labels_tensor
    dataset = CustomSequenceDataset(sentences, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=shuffle)

    return loader
            
def get_attribution_dataloader(paths, species, vocab):
    all_sentences = []
    all_species = []
    data = zip(paths, species)
    for path, s in data:
        with open(path, encoding="utf-8") as f:
                content = f.read()
        genome = tokenization(content)
        genome = [i for i in genome if i !='Cluster']
        sentence = []
        for gene in genome:
            try:
                sentence.append(vocab[gene])
            except:
                continue
        if len(sentence) > 100:
            all_sentences.append(sentence)
            all_species.append(s)

    label_encoder = LabelEncoder()
    species_encoded = label_encoder.fit_transform(all_species)
    species_encoded = torch.from_numpy(species_encoded)

    def collate_fn(batch):
        sentences = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        labels = [item[1] for item in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=vocab['<PAD>'])
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        return padded_sentences, labels_tensor

    dataset = CustomSequenceDataset(all_sentences, species_encoded)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    return dataloader, label_encoder


def interpret_sentence(model, sentence, token_reference, lig, i2w, device):
    model.zero_grad()
    input_indices = sentence.clone().detach().to(device).long().unsqueeze(0)
    seq_length = len(sentence)
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    attributions_ig, delta = lig.attribute(input_indices, reference_indices.long(), n_steps=100, return_convergence_delta=True)
    text = [i2w[token.item()] for token in sentence]
    attributions = attributions_ig.squeeze()
    attributions = attributions.sum(dim=1).squeeze()
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    return text, attributions

class WrappedModel(torch.nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        x_int = x.long()
        return self.original_model(x_int)
    
def main():
    parser = argparse.ArgumentParser(description='Compute integrated gradients for GELATO model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to attribution result')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--vocab_dir', type=str, required=True, help='Path to vocab directory')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--c', type=str, required=True, help='Compounds requiring attribution analysis')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")
    gpus = [device]
    compound = args.c
    
    vocab_path = f'{args.vocab_dir}/{compound}_w2i.json'
    model_path = f'{args.model_dir}/{compound}_model.pth'
    w2i = json.load(open(vocab_path, 'r'))
    i2w = {v: k for k, v in w2i.items()}
    model = GELATO(True, w2i, 50, 200, [10], 1, 0.7)
    model.load_state_dict(torch.load(model_path))
    model.to(gpus[0])
    model.eval()
    train_data = pd.read_csv(f'{args.data_dir}/train_data_{compound}.csv')
    test_data = pd.read_csv(f'{args.data_dir}/test_data_{compound}.csv')
    valid_data = pd.read_csv(f'{args.data_dir}/valid_data_{compound}.csv')
    all_data = pd.concat([train_data, test_data, valid_data])
    positive_df = all_data[all_data[compound]==1]
    labels = positive_df[compound].tolist()
    paths = positive_df['path'].tolist()
    
    drop_paths = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            content = f.read()
        genome = tokenization(content)
        genome = [i for i in genome if i !='Cluster']
        sentence = []
        for gene in genome:
            try:
                sentence.append(w2i[gene])
            except:
                continue
        if len(sentence) <= 100:
            drop_paths.append(p)
    positive_df = positive_df[~positive_df['path'].isin(drop_paths)]
    positive_dataloader = get_dataloader(positive_df['path'].tolist(), labels, w2i, 1)
    predictions = []
    for X, y in positive_dataloader:
        with torch.no_grad():
            X = X.to(gpus[0])
            labels = y.cpu()
            pred = model(X)
            predicts = torch.round(torch.sigmoid(pred)).cpu()
            for p in predicts:
                predictions.append(p.item())
    positive_df["pred"] = predictions
    filtered_positive_df = positive_df[positive_df["pred"]==positive_df[compound]]
    file_path = filtered_positive_df["path"].tolist()
    species = filtered_positive_df["Species"].tolist()
    correct_dataloader, label_encoder = get_attribution_dataloader(file_path, species, w2i)
    
    # get attributions with correct species
    PAD_IND = w2i["<PAD>"]
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(model, model.embedding)
    wrapped_model = WrappedModel(model.to(gpus[0]))
    value_result = {}
    for sentences, species in correct_dataloader:
        for i in range(sentences.size(0)):
            sentence = sentences.to(gpus[0])[i,:]
            species_label = species[i]
            decoded = label_encoder.inverse_transform([int(species_label.item())])
            text, word_attributions = interpret_sentence(wrapped_model, sentence, token_reference, lig, i2w, gpus[0])
            org_text_value = dict(zip(text, word_attributions))
            value_result[decoded.item()] = org_text_value
            
    with open(f'{args.output_dir}/{compound}_value_result.json', 'w') as file:
        json.dump(value_result, file)


    penc = 95
    for root, dirs, files in os.walk(f'{args.output_dir}'):
        for file in files:
            try:
                dic = json.load(open(os.path.join(root, file), 'r'))
                
                pos_vote_result = defaultdict(int)
                neg_vote_result = defaultdict(int)
                pos_ranks = defaultdict(list)
                neg_ranks = defaultdict(list)
                N = len(dic)
                for s, d in dic.items():
                    genes = list(d.keys())
                    values = np.array(list(d.values()))
                    upper_percentile = np.percentile(values, penc)
                    lower_percentile = np.percentile(values, 100 - penc)
                    df = pd.DataFrame({'gene': genes, 'value': values})
                    pos_df = df[(df['value'] > upper_percentile) & (df['value'] > 0)].copy()
                    if not pos_df.empty:
                        pos_df = pos_df.sort_values('value', ascending=False)
                        pos_df['rank_pos'] = pos_df['value'].rank(ascending=False, method='min')  
                        for _, row in pos_df.iterrows():
                            gene = row['gene']
                            pos_vote_result[gene] += 1
                            pos_ranks[gene].append(row['rank_pos'])
                    
                    neg_df = df[(df['value'] < lower_percentile) & (df['value'] < 0)].copy()
                    if not neg_df.empty:
                        neg_df = neg_df.sort_values('value', ascending=True)  
                        neg_df['rank_neg'] = neg_df['value'].rank(ascending=True, method='min')
                        for _, row in neg_df.iterrows():
                            gene = row['gene']
                            neg_vote_result[gene] += 1
                            neg_ranks[gene].append(row['rank_neg'])

                cutoff = int(N * 0.05) if N > 0 else 1  
                pos_vote = {k: v for k, v in pos_vote_result.items() if v >= cutoff}
                neg_vote = {k: v for k, v in neg_vote_result.items() if v >= cutoff}
                pos_genes = list(pos_vote.keys())
                neg_genes = list(neg_vote.keys())
                intersect_genes = list(set(pos_genes) & set(neg_genes))
                pos_genes = [x for x in pos_genes if x not in intersect_genes]
                neg_genes = [x for x in neg_genes if x not in intersect_genes]
                
                pos_stats = []
                for gene in pos_genes:
                    freq = pos_vote[gene] / N if N > 0 else 0
                    avg_rank = np.mean(pos_ranks[gene]) if pos_ranks[gene] else float('inf')
                    score = freq / avg_rank if avg_rank != 0 else 0
                    pos_stats.append({'gene': gene, 'pos_freq': freq, 'average_rank_pos': avg_rank, 'pos_score': score})
                
                neg_stats = []
                for gene in neg_genes:
                    freq = neg_vote[gene] / N if N > 0 else 0
                    avg_rank = np.mean(neg_ranks[gene]) if neg_ranks[gene] else float('inf')
                    score = freq / avg_rank if avg_rank != 0 else 0
                    neg_stats.append({'gene': gene, 'neg_freq': freq, 'average_rank_neg': avg_rank, 'neg_score': score})

                pos_stats_df = pd.DataFrame(pos_stats).sort_values('pos_score', ascending=False)
                neg_stats_df = pd.DataFrame(neg_stats).sort_values('neg_score', ascending=False)

                compound_name = os.path.splitext(file)[0]
                pos_stats_df.to_csv(f'{args.output_dir}/positive_gene_importance_{compound_name}.csv', index=False)
                neg_stats_df.to_csv(f'{args.output_dir}/negative_gene_importance_{compound_name}.csv', index=False)
                pos_dict = dict(zip(pos_stats_df['gene'].tolist(), pos_stats_df['pos_score'].tolist()))
                neg_dict = dict(zip(neg_stats_df['gene'].tolist(), neg_stats_df['neg_score'].tolist()))
                dic = {
                    'pos_hif':pos_dict,
                    'neg_hif':neg_dict
                }
                with open(f'{args.output_dir}/{compound_name}_KAG.json', 'w') as f:
                    json.dump(dic, f)
            except:
                print(f'Error: {file}')

if __name__ == "__main__":
    main()