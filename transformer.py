import torch
import tiktoken as ttk
from torch.nn import functional as F


class EmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # define some important vars
        self.vocab_size = 0
        self.data = None
        self.process_input()

        # FIXME im losing my shti i need to find a better way to format this so that we can get the data in a little better
        self.train_data = data[:n]

        # determines how much of the data from the training set the model will see at a time
        context_size = 10

        # how many pieces of data are we sending off for the gpu to process in parallel
        batch_size = 4

        self.token_embedding_table = torch.nn.Embedding(
            self.vocab_size, self.vocab_size
        )

    def forward(self, x, y):
        logits = self.token_embedding_table(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def get_batch(self, split="train"):
        data = train

    def process_input(self):
        with open("input.txt", "r") as f:
            text = f.read()

            # this might be reflective of the encoder model but for right now i dont actually know
            self.vocab_size = len(set(text))

            # tokenize with byte pair encoding.
            # This gives us a shorter token array lenght becuase we arent splitting by character
            enc = ttk.get_encoding("gpt2")
            enc.n_vocab
            encoded = enc.encode(text)
            self.data = torch.tensor(encoded, dtype=torch.long)
            # looking at the tokenized output will essentially give us a "one to one" translation of the text


def main():

    embedding_layer = EmbeddingLayer()
    logits, loss = embedding_layer.forward()
    print(loss)


if __name__ == "__main__":
    main()
