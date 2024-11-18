import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def simple_tokenizer(text):
    text = text.lower().split()
    
    return text

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_count = 4 

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.word_count
            self.idx2word[self.word_count] = word
            self.word_count += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<UNK>"]
        return self.word2idx[word]

class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, max_length=512):
        self.articles = articles
        self.summaries = summaries
        self.max_length = max_length
        self.vocab = Vocabulary()
        self.build_vocab()
        
    def build_vocab(self):

        for article in self.articles:
            for word in simple_tokenizer(article):
                self.vocab.add_word(word)
                
        for summary in self.summaries:
            for word in simple_tokenizer(summary):
                self.vocab.add_word(word)
    
    def __len__(self):
        return len(self.articles)
    
        
    def __getitem__(self, idx):

        
        article = self.articles[idx]
        summary = self.summaries[idx]


        article_tokens = simple_tokenizer(article)
        summary_tokens = simple_tokenizer(summary)


        article_indices = [self.vocab(word) for word in article_tokens]
        summary_indices = [self.vocab(word) for word in summary_tokens]


        article_indices = self._pad_sequence(article_indices)
        summary_indices = self._pad_sequence(summary_indices)

        return {
            'article': torch.tensor(article_indices),
            'summary': torch.tensor(summary_indices)
        }
        
    def _pad_sequence(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence + [self.vocab.word2idx["<PAD>"]] * (self.max_length - len(sequence))
        
        
        
        
        