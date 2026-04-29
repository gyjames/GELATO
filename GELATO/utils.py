import os
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def extract_name(filepath):
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]
    return name

def tokenization(text):
    filters = ['!', '\"', '#', '$', '%', '&', '\\(', '\\)', '\\*', '\\+', ',', '-', '/', ':', ';', '<', '=', '>','\\?', '@', '\\[', '\\\\', '\\]', '^', '_', '`', '\\{', '\\|', '\\}', '~', '\\t', '\\n', '\\x97', '\\x96', '”', '“']
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(filters), " ", text, flags=re.S)
    ls = [i.strip() for i in text.split()]
    for i, w in enumerate(ls):
        w = re.sub(r'\.$', '', w)
        ls[i] = w
    return ls

def get_file_paths(folder_path):
    file_paths = []
    query_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            query_names.append(file)
    return file_paths, query_names

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = json.load(f)
    return vocab

def merge(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict

def min_max_normalize(d, new_min=0, new_max=1):
    values = list(d.values())
    if len(set(values)) == 1:
        return {k: new_min for k in d}
    min_val = min(values)
    max_val = max(values)
    normalized_dict = {k: (v - min_val) / (max_val - min_val) * (new_max - new_min) + new_min for k, v in d.items()}
    return normalized_dict

class SequenceDataset(Dataset):
    def __init__(self, sequences, pad_token=1):
        self.sequences = sequences
        self.pad_token = pad_token

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)

def build_dataloader(sentences_path, vocab, batch_size=1):
    def collate_fn(batch, min_length=10, pad_token=1):
        sentences = [item for item in batch]
        sentences = [
            item if len(item) >= min_length 
            else torch.cat([item, torch.full((min_length - len(item),), pad_token, dtype=torch.long)]) 
            for item in sentences
        ]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_token)
        return padded_sentences
    
    sentences = []
    for path in sentences_path:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        genome = tokenization(content)
        genome = [i for i in genome if i != 'Cluster']
        sentence = []
        for gene in genome:
            try:
                sentence.append(vocab[gene])
            except:
                continue
        sentences.append(sentence)

    dataset = SequenceDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return loader