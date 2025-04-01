import torch
import tiktoken as ttk
from torch.nn import functional as F


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, data):
        super().__init__()

        # define some important vars
        self.vocab_size = vocab_size
        self.data = data

        # Going to assume the dimension is going to be 5
        self.token_embedding_table = torch.nn.Embedding(
            self.vocab_size, 10
        )

    def forward(self, x, y: torch.Tensor = None):
        logits: torch.Tensor = self.token_embedding_table(x)
        with open("help.txt", "w") as f:
            for _ in logits.tolist():
                for i in _:
                    f.write(str(i) + "\n")

        from pprint import pprint
        pprint(str(logits.tolist()) + "\n\n\n\n")
        pprint(x)
        exit()
        if y is None:
            loss = None
        else:
            # logits becomesa tensor of size (Batch size, Sequence Length (T), vocab_size)
            B, T, C  = logits.shape  # (Batch size, Sequence Length (T), vocab_size)
            logits = logits.view(
                B * T, C
            )  # reshape the logits so they can be used in cross entropy loss
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self.forward(x)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1)  # (B, T+1)
        return x


def process_input(file_name):
    with open(file_name, "r") as f:
        text = f.read()

        # this might be reflective of the encoder model but for right now i dont actually know
        vocab_size = len(set(text))

        # tokenize with byte pair encoding.
        # This gives us a shorter token array lenght becuase we arent splitting by character
        enc = ttk.get_encoding("gpt2")
        encoded = enc.encode(text)
        return enc.n_vocab, torch.tensor(encoded, dtype=torch.long)
        # looking at the tokenized output will essentially give us a "one to one" translation of the text


def get_batch(split, train_data, val_data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def main():

    vocab_size, data = process_input("input.txt")
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    # block size and batch size can change
    xb, yb = get_batch("train", train_data, val_data, 10, 5)

    embedding_layer = EmbeddingLayer(vocab_size, data)
    _, loss = embedding_layer.forward(xb, yb)
    print(loss)

    enc = ttk.get_encoding("gpt2")

    decoded = enc.decode(
        embedding_layer.generate(
            torch.zeros(1, 1, dtype=torch.long), max_new_tokens=100
        )[0].tolist()
    )
    print(decoded)
    # could do SGD but whatever
    optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=1e-3)

    # train the model
    for steps in range(100):
        xb, yb = get_batch("train", train_data, val_data, 10, 5)
        logits, loss = embedding_layer.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(loss.item())


if __name__ == "__main__":
    main()
