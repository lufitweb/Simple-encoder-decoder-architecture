from torch import nn
import torch
import torch.nn.functional as F
from random import random
import random



class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        
    def forward(self, x):
        embedded = self.embedding(x)
                
        outputs, (hidden, cell) = self.lstm(embedded) 
        return outputs, (hidden, cell)  
        
        
        
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_word, context, hidden_state):

        input_word = input_word.unsqueeze(1) 
        

        embedded = self.embedding(input_word)

        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        lstm_input = torch.cat([embedded, context], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, hidden_state)
        
        prediction = self.output_layer(output)
        
        return prediction, (hidden, cell)
        
        
        
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.align = nn.Linear(hidden_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_length, 1)
        
        combined = torch.cat((encoder_outputs, decoder_hidden), dim=2)
        
        energy = self.attention(combined)
        attention_scores = self.align(energy)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.transpose(1,2), encoder_outputs)
        
        return context, attention_weights
            
        
        
class Summarizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        max_len = target.size(1)
        vocab_size = self.decoder.output_layer.out_features

        outputs = torch.zeros(batch_size, max_len, vocab_size)

        encoder_outputs, (hidden, cell) = self.encoder(source)

        decoder_input = target[:, 0]

        for t in range(1, max_len):
            
            context, weights = self.attention(hidden[-1], encoder_outputs)

            decoder_output, (hidden, cell) = self.decoder(
                decoder_input, context, (hidden, cell))
            
            outputs[:, t] = decoder_output.squeeze(1)  # Add squeeze here!

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, t] if teacher_force else decoder_output.argmax(2).squeeze(1)

        return outputs
    
        
        
        
        