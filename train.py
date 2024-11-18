import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasex import SummarizationDataset
from model import Summarizer

def main():

    train_articles = [
        "The cat sat on the mat",
        "Dogs love playing in the park"
    ]
    
    train_summaries = [
        "Cat sat on mat",
        "Dogs play in park"
    ]


    vocab_size = 1000  
    embedding_dim = 256
    hidden_dim = 512
    

    train_dataset = SummarizationDataset(
        articles=train_articles,
        summaries=train_summaries
    )
    
    #create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,
        shuffle=True
    )

    #initialize model
    model = Summarizer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    pad_idx = 0 

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters())

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            source = batch['article']
            target = batch['summary']
            
            output = model(source, target)
            
            output = output.view(-1, vocab_size)
            target = target.view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
    print("Training finished! Saving model...")
# Save the complete model state
    torch.save({
        'encoder': {
            'embedding.weight': model.encoder.embedding.weight,
            'lstm.weight_ih_l0': model.encoder.lstm.weight_ih_l0,
            'lstm.weight_hh_l0': model.encoder.lstm.weight_hh_l0,
            'lstm.bias_ih_l0': model.encoder.lstm.bias_ih_l0,
            'lstm.bias_hh_l0': model.encoder.lstm.bias_hh_l0
        },
        'decoder': {
            'embedding.weight': model.decoder.embedding.weight,
            'lstm.weight_ih_l0': model.decoder.lstm.weight_ih_l0,
            'lstm.weight_hh_l0': model.decoder.lstm.weight_hh_l0,
            'lstm.bias_ih_l0': model.decoder.lstm.bias_ih_l0,
            'lstm.bias_hh_l0': model.decoder.lstm.bias_hh_l0,
            'output_layer.weight': model.decoder.output_layer.weight,
            'output_layer.bias': model.decoder.output_layer.bias
        },
        'attention': {
            'attention.weight': model.attention.attention.weight,
            'attention.bias': model.attention.attention.bias,
            'align.weight': model.attention.align.weight,
            'align.bias': model.attention.align.bias
        }
        }, 'summarizer_model.pth')

    print("Model saved to checkpoint.pth")

if __name__ == "__main__":
    main()