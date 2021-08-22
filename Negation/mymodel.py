import torch.nn as nn
import torch
from transformers import BertForTokenClassification, BertTokenizer, BatchEncoding, pipelines
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.modules import CrossEntropyLoss
from sklearn.metrics import recall_score, precision_score, f1_score
import csv
import json
from tqdm import tqdm
PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'
import numpy as np
import nltk.tokenize as tk

def pad_sequence(x, max_len, type=np.int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x

def my_collate(x):

    words = [x_['tokens_id'] for x_ in x]
    labels = [x_['label'] for x_ in x]
    tokens = [x_['tokens'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)
    labels = pad_sequence(labels, max_seq_len)

    return inputs_id, labels

def my_collate_test(x):

    words = [x_['tokens_id'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)

    return inputs_id, seq_len

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def get_region(text_raw, tokens):
    initial_id = 0
    text_raw = text_raw.lower()
    results = []
    for token in tokens:
        text = text_raw[initial_id:]
        for cid, char in enumerate(token):
            if char!='#':
                break
        token = token[cid:]
        s_pos = initial_id + text.find(token)
        e_pos = s_pos + len(token)
        results.append((s_pos, e_pos))
        initial_id = e_pos
    return results


def data_process(data_path):
    tokenizer = BertTokenizer.from_pretrained('./bert_medical_large')
    instances = []
    data_file = open(data_path, 'r', encoding='UTF-8')
    data_reader = csv.reader(data_file)
    next(data_reader)
    for row in tqdm(data_reader):
        text = row[0]
        tokens = tokenizer.tokenize(text)
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        tokens_position = get_region(text, tokens)
        scope = json.loads(row[3])
        labels = np.zeros(len(tokens))
        if row[3] != 'NaN':
            for k in range(len(tokens)):
                if tokens_position[k][0] >= scope[0] and tokens_position[k][1] <= scope[1]:
                    labels[k] = 1

        dict_instance = {'label': labels,
                        'tokens': tokens,
                        "tokens_id": tokens_id,
                        "tokens_pos": tokens_position}

        instances.append(dict_instance)

    return instances

class ScopeModel:
    def __init__(self, full_finetuning=True, train=False, pretrained_model_path='Scope_Resolution_Augment.pickle',
                 device='cuda', learning_rate=1e-5, gpu=0, bert_path='f:/caml-mimic-master/Negation/bert_medical_large'):
        self.gpu = gpu
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model_name = bert_path

        self.num_labels = 2
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model_path = pretrained_model_path
        if train == True:
            self.model = BertForTokenClassification.from_pretrained(self.bert_path, num_labels=self.num_labels)

        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device == 'cuda':
            self.model.cuda(gpu)
        else:
            self.model.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    def train(self, train_data, val_data):
        loss_fn = CrossEntropyLoss()
        max_score= 0
        for e in range(30):
            print('Start Epoch %d' %(e))

            train_loader = DataLoader(MyDataset(train_data), 16, shuffle=True, collate_fn=my_collate)
            data_iter = iter(train_loader)
            num_iter = len(train_loader)
            for i in tqdm(range(num_iter)):
                inputs_id, labels = next(data_iter)
                inputs_id, labels = torch.LongTensor(inputs_id), torch.LongTensor(labels)
                if not(self.gpu is None):
                    inputs_id, labels = inputs_id.cuda(self.gpu), labels.cuda(self.gpu)
                logits = self.model(inputs_id)[0]
                loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print("loss: %f" %(loss.item()))
            eval_score = self.eval(val_data)
            if eval_score > max_score:
                max_score = eval_score
                torch.save(self.model, self.model_path)
                print('Best Epoch:%d, score:%f' %(e, eval_score))


    def eval(self, val_data):
        val_loader = DataLoader(MyDataset(val_data), 12, shuffle=False, collate_fn=my_collate)
        data_iter = iter(val_loader)
        num_iter = len(val_loader)
        self.model.eval()
        predictions = []
        y_truth = []
        for i in tqdm(range(num_iter)):
            inputs_id, labels = next(data_iter)
            inputs_id, labels = torch.LongTensor(inputs_id), torch.LongTensor(labels)
            if not (self.gpu is None):
                inputs_id, labels = inputs_id.cuda(self.gpu), labels.cuda(self.gpu)
            logits = self.model(inputs_id)[0]
            pre_y = torch.softmax(logits, dim=2).view(-1, 2)
            pre_y = pre_y.cpu().detach().numpy()
            pre_y = np.argmax(pre_y, axis=1)
            labels = labels.view(-1)
            labels = labels.cpu().detach().numpy()
            predictions.append(pre_y)
            y_truth.append(labels)
        predictions = np.concatenate(predictions, axis=0)
        y_truth = np.concatenate(y_truth, axis=0)
        score = f1_score(y_truth, predictions)
        self.model.train()
        return score

    def predict(self, text):
        self.model.eval()
        tokens = self.tokenizer.tokenize(text)
        num_sentences = len(tokens)//512
        inputs_id = []
        for nid in range(num_sentences+1):
            sentence = tokens[nid*512:(nid+1)*512]
            input_id = self.tokenizer.convert_tokens_to_ids(sentence)
            inputs_id.append(input_id)
        seq_len = [len(w) for w in inputs_id]
        max_seq_len = max(seq_len)
        inputs_id = pad_sequence(inputs_id, max_seq_len)
        inputs_id = torch.LongTensor(inputs_id)
        if self.device.type=='cuda':
            inputs_id = inputs_id.cuda(self.gpu)

        logits = self.model(inputs_id)[0]
        logits = logits.cpu().detach().numpy()
        labels = np.argmax(logits, axis=2)
        new_labels = []
        for label, length in zip(labels, seq_len):
            new_labels.append(label[0:length])
        labels = np.concatenate(new_labels, axis=0)
        for tid in range(len(tokens)):
            tokens[tid] = tokens[tid].replace('#', '')
        return labels, tokens












