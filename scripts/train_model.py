import os
import re
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



def extract_name(filepath):
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]
    return name

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

def train(model, iterator, optimizer, criterion, gpus):
    epoch_loss = 0
    metabolism_labels = []
    predictions_ls = []
    probabilities_ls = []
    model.train()
    for X, y in iterator:
        X, y = X.to(gpus[0]), y.to(gpus[0])
        y = y.float()
        optimizer.zero_grad()
        predictions = model(X).squeeze(1)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        labels = y.cpu()
        for i in labels:
            metabolism_labels.append(i.item())
        probabilities = torch.sigmoid(predictions)
        predictions = torch.round(probabilities).cpu()
        for i in predictions:
            predictions_ls.append(i.item())
        probabilities = probabilities.cpu()
        for i in probabilities:
            probabilities_ls.append(i.item())
    epoch_loss /= len(iterator)
    acc = metrics.accuracy_score(metabolism_labels, predictions_ls)
    fpr, tpr, thresholds = metrics.roc_curve(metabolism_labels, probabilities_ls)
    auc = metrics.auc(fpr, tpr)        
        
    return epoch_loss, acc, auc

def evaluate(model, iterator, criterion, gpus):
    epoch_loss = 0
    metabolism_labels = []
    predictions_ls = []
    probabilities_ls = []
    model.eval()
    with torch.no_grad():
        for X, y in iterator:
            X, y = X.to(gpus[0]), y.to(gpus[0])
            y = y.float()
            predictions = model(X).squeeze(1)
            loss = criterion(predictions, y)
            epoch_loss += loss.item()
            labels = y.cpu()
            for i in labels:
                metabolism_labels.append(i.item())
            probabilities = torch.sigmoid(predictions)
            predictions = torch.round(probabilities).cpu()
            for i in predictions:
                predictions_ls.append(i.item())
            probabilities = probabilities.cpu()
            for i in probabilities:
                probabilities_ls.append(i.item())
    epoch_loss /= len(iterator)
    acc = metrics.accuracy_score(metabolism_labels, predictions_ls)
    fpr, tpr, thresholds = metrics.roc_curve(metabolism_labels, probabilities_ls)
    auc = metrics.auc(fpr, tpr)   
    
    return epoch_loss, acc, auc

def train_regression(model, iterator, optimizer, criterion, gpus):
    epoch_loss = 0
    target = []
    pred_ls = []
    model.train()
    for batch_idx, (X, y) in enumerate(iterator):
        X, y = X.to(gpus[0]), y.to(gpus[0])
        optimizer.zero_grad()
        pred = model(X).squeeze(1)
        loss = criterion(pred, y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        for i in y:
            target.append(i.item())
        for i in pred:
            pred_ls.append(i.item())

    r2 = metrics.r2_score(target, pred_ls)
    mse = metrics.mean_squared_error(target, pred_ls)
    rmse = np.sqrt(mse)
    return epoch_loss / len(iterator), r2, rmse

def evaluate_regression(model, iterator, criterion, gpus):
    epoch_loss = 0
    target = []
    pred_ls = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(iterator):
            X, y = X.to(gpus[0]), y.to(gpus[0])
            pred = model(X).squeeze(1)
            loss = criterion(pred, y.float())
            epoch_loss += loss.item()
            for i in y:
                target.append(i.item())
            for i in pred:
                pred_ls.append(i.item())
    r2 = metrics.r2_score(target, pred_ls)
    mse = metrics.mean_squared_error(target, pred_ls)
    rmse = np.sqrt(mse)
    return epoch_loss / len(iterator), r2, rmse

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_better_assembly_genome(df, top_n=1):
    df.loc[:, "assembly_level"] = df["assembly_level"].replace({"Contig": 3, "Scaffold": 2, "Chromosome": 1, "Complete Genome": 0})
    df_sorted = df.sort_values(by=['Species', 'assembly_level'])
    df = df_sorted.groupby('species_taxid').head(top_n)
    return df


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

class GELATO_r(nn.Module):
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

        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, dropout=0.3, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (filter_sizes[0], embedding_dim))
        self.fc = nn.Linear(n_filters, 50)
        self.fc_out = nn.Linear(50, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _create_padding_mask(self, text):
        mask = (text == self.padding_idx)
        return mask
    
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

def tokenization(text):
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '/', ':', ';', '<', '=', '>','\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    ls = [i.strip() for i in text.split()]
    for i, w in enumerate(ls):
        w = re.sub(r'\.$', '', w)
        ls[i] = w
    return ls

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

class BuildVocab:
    def __init__(self, data_path=None, min_word_occurences=5, max_word_occurences=None, oov_token='<oov>', pad_token="<PAD>", max_feature=None):
        self.total_file_path = data_path
        self.word_counter = {}
        self.w2i = {pad_token:0, oov_token: 1}
        self.genome_len = []
        for file_path in self.total_file_path:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            genome = tokenization(content)
            genome = [i for i in genome if i !='Cluster']
            self.genome_len.append(len(genome))
            for word in genome:
                if word not in self.word_counter:
                    self.word_counter[word] = 0
                self.word_counter[word] += 1
        if max_word_occurences is not None:
            self.word_counter = {k: v for k, v in self.word_counter.items() if v <= max_word_occurences}
        if isinstance(max_feature, int):
            temp = sorted(list(self.word_counter.items()), key=lambda x: x[1], reverse=True)[:max_feature]
            self.word_counter = dict(temp)
        else:
            temp = sorted(list(self.word_counter.items()), key=lambda x: x[1], reverse=True)
            self.word_counter = dict(temp)
        for w, c in self.word_counter.items():
            if c >= min_word_occurences:
                self.w2i[w] = len(self.w2i)
        self.i2w = dict(zip(self.w2i.values(), self.w2i.keys()))
        self.seq_len = sum(self.genome_len) / len(self.genome_len)

def main():
    parser = argparse.ArgumentParser(description='Train GELATO model for compound prediction')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model folder')
    parser.add_argument('--type', type=str, required=True, help='Type of model')
    parser.add_argument('--train', type=str, required=True, help='Path to train data')
    parser.add_argument('--valid', type=str, required=True, help='Path to valid data')
    parser.add_argument('--test', type=str, required=True, help='Path to test data')
    parser.add_argument('--c', type=str, required=True, help='Phenotype')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    gpus = [device]
    test_data = pd.read_csv(args.test)
    valid_data = pd.read_csv(args.valid)
    train_data = pd.read_csv(args.train)
    train_labels = train_data[args.c].tolist()
    test_labels = test_data[args.c].tolist()
    valid_labels = valid_data[args.c].tolist()
    train_paths = train_data['path'].tolist()
    valid_paths = valid_data['path'].tolist()
    test_paths = test_data['path'].tolist()
    vocab = BuildVocab(train_paths+valid_paths, max_word_occurences=len(train_paths+valid_paths))
    w2i = vocab.w2i
    os.makedirs(f'{args.model_dir}/vocab', exist_ok=True)
    os.makedirs(f'{args.model_dir}/model', exist_ok=True)
    with open(f'{args.model_dir}/vocab/{args.c}_w2i.json', 'w') as f:
        json.dump(w2i, f)
    train_dataloader = get_dataloader(train_paths, train_labels, w2i, BATCH_SIZE=16, shuffle=True)
    valid_dataloader = get_dataloader(valid_paths, valid_labels, w2i, BATCH_SIZE=1, shuffle=True)
    test_dataloader = get_dataloader(test_paths, test_labels, w2i, BATCH_SIZE=1)
    
    if args.type == 'c':
        result = []
        model = GELATO(True, w2i, 50, 200, [10], 1, 0.7)
        optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        model = model.to(gpus[0])
        criterion = criterion.to(gpus[0])
        cur_best_auc = float('-inf')
        cur_best_acc = float('-inf')
        pre_loss = float('inf')
        patience = 0
        for epoch in range(200):
            start_time = time.time()
            train_loss, train_acc, train_auc = train(model, train_dataloader, optimizer, criterion, gpus)
            val_loss, val_acc, val_auc = evaluate(model, valid_dataloader, criterion, gpus)
            scheduler.step()
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.5f} | Train Auc: {train_auc*100:.3f}%  | Train Acc: {train_acc*100:.3f}%')
            print(f'\t Val. Loss: {val_loss:.5f} |  Val. Auc: {val_auc*100:.3f}%  |  Val. Acc: {val_acc*100:.3f}%')
            if val_acc > cur_best_acc or (abs(val_acc - cur_best_acc) < 1e-4 and val_auc > cur_best_auc):
                cur_best_acc = val_acc
                cur_best_auc = val_auc
                torch.save(model.state_dict(), f'{args.model_dir}/model/{args.c}_model.pth')
                print(f'Epoch {epoch+1}: model saved')
            if val_loss > pre_loss:
                patience += 1
                if patience >= 5:
                    print("EARLY STOPPING TRIGGERED")
                    break   
            else:
                patience = 0
            pre_loss = val_loss

        model = GELATO(False, w2i, 50, 200, [10], 1, 0.7)
        model.to(gpus[0])
        model.load_state_dict(torch.load(f'{args.model_dir}/model/{args.c}_model.pth'))
        model.eval()
        metabolism_labels = []
        predictions_ls = []
        probabilities_ls = []
        model.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(gpus[0]), y.to(gpus[0])
                predictions = model(X).squeeze(1)
                labels = y.cpu()
                for i in labels:
                    metabolism_labels.append(i.item())
                probabilities = torch.sigmoid(predictions)
                predictions = torch.round(probabilities).cpu()
                for i in predictions:
                    predictions_ls.append(i.item())
                probabilities = probabilities.cpu()
                for i in probabilities:
                    probabilities_ls.append(i.item())
        acc = metrics.accuracy_score(metabolism_labels, predictions_ls)
        pre = metrics.precision_score(metabolism_labels, predictions_ls)
        recall = metrics.recall_score(metabolism_labels, predictions_ls)
        f1 = metrics.f1_score(metabolism_labels, predictions_ls)
        fpr, tpr, thresholds = metrics.roc_curve(metabolism_labels, probabilities_ls)
        auc = metrics.auc(fpr, tpr)
        n = [args.c, acc, pre, recall, f1, auc]
        result.append(n)
        result_df = pd.DataFrame(result, columns=["Compound", 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'])
        result_df.to_csv(f"{args.model_dir}/{args.c}_result.csv", index=False)
    
    elif args.type == 'r':
        result = []
        model = GELATO_r(True, w2i, 300, 200, [10], 1, 0.3)
        optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
        criterion = nn.MSELoss()
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        model = model.to(gpus[0])
        criterion = criterion.to(gpus[0])
        cur_best_r2 = float('-inf')
        pre_loss = float('inf')
        patience = 0
        train_loss_ls = []
        train_r2_ls = []
        val_loss_ls = []
        val_r2_ls = []
        for epoch in range(200):
            start_time = time.time()
            train_loss, train_r2, train_rmse = train_regression(model, train_dataloader, optimizer, criterion, gpus)
            valid_loss, valid_r2, valid_rmse = evaluate_regression(model, valid_dataloader, criterion, gpus)
            train_loss_ls.append(train_loss)
            train_r2_ls.append(train_r2)
            val_loss_ls.append(valid_loss)
            val_r2_ls.append(valid_r2)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train R2: {train_r2:.3f} | Train RMSE: {train_rmse:.3f}')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid R2: {valid_r2:.3f} | Valid RMSE: {valid_rmse:.3f}')
            if valid_r2 > cur_best_r2:
                cur_best_r2 = valid_r2
                torch.save(model.state_dict(), f'{args.model_dir}/model/{args.c}_model.pth')
            if valid_loss > pre_loss:
                patience += 1
                if patience >= 5:  
                    print("EARLY STOPPING TRIGGERED")
                    break   
            else:
                patience = 0
            pre_loss = valid_loss

        model = GELATO_r(False, w2i, 300, 200, [10], 1, 0.3)
        model.to(gpus[0])
        model.load_state_dict(torch.load(f'{args.model_dir}/model/{args.c}_model.pth'))
        model.eval()
        target= []
        pred_ls = []

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_dataloader):
                X, y = X.to(gpus[0]), y.to(gpus[0])
                pred = model(X).squeeze(1)
                pred = pred.cpu().detach().numpy() 
                for i in y:
                    target.append(i.item())
                for i in pred:
                    pred_ls.append(i.item())
            
        r2 = metrics.r2_score(target, pred_ls)
        mse = metrics.mean_squared_error(target, pred_ls)
        rmse = np.sqrt(mse)
        n = [args.c, r2, rmse]
        result.append(n)
        result_df = pd.DataFrame(result, columns=["Compound", 'R2', 'RMSE'])
        result_df.to_csv(f"{args.model_dir}/{args.c}_result.csv", index=False)

if __name__ == "__main__":
    main()