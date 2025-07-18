import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def get_embeddings():
    vocab_size = 50257
    output_dim = 256
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #initialized randomly
    
    max_length = 4
    context_length = max_length
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )
    token_embeddings = token_embedding_layer(dataloader.input_ids) #Apply embeddings to a list of token ids (look-up operation)
    
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) #Absolute positional embeddings, one row for each posible position of a token in the context 
    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) #input is a placeholder

    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings

