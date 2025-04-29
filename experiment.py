import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os
import sys
from custom_dataloader import CBOWDataset

print("starting...", file=sys.stderr)

# hyperparameters
BATCH_SIZE = 64  # how many independent sequences will we process in parallel?
CONTEXT_SIZE = 4  # what is the maximum context length for predictions?
D_EMBEDDINGS = 20

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4

print(torch.cuda.is_available())
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


class EmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
    ):
        super().__init__()

        # define some important vars
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding_table = torch.nn.Embedding(
            vocab_size,
            d_model,
        )
        self.linear_one = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embeds = self.token_embedding_table(x)
        input_embeds = embeds.sum(dim=1)  # (batch_size, d_model)
        out = self.linear_one(input_embeds)  # (batch_size, vocab_size)
        return out


def process_input(tokenizer, num_proc):

    ds = load_dataset("Publishing/pretraining_v1", split="train")
    # ds = load_dataset(
    #     "wikimedia/wikipedia",
    #     "20231101.en",
    #     split="train",
    #     cache_dir="/scratch",
    #     num_proc=num_proc,
    # )
    # when we pad now the map function will know what to do
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized = ds.map(tokenize_function, batched=True, num_proc=num_proc)

    dataset = CBOWDataset(tokenized, context_size=2)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader


def build_cbow_pairs(tokens, context_size=2):
    pairs = []
    for i in range(context_size, len(tokens) - context_size):
        left = tokens[i - context_size : i]
        right = tokens[i + 1 : i + context_size + 1]
        context = left + right  # still list of ints
        center = tokens[i]
        pairs.append((context, center))
    return pairs


def main():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_proc = int(os.cpu_count() / 2)

    dataloader = process_input(tokenizer, num_proc)

    vocab_size = tokenizer.vocab_size
    embedding_layer = EmbeddingLayer(vocab_size, D_EMBEDDINGS)
    embedding_layer.to("cuda")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=1e-3)

    # train the model
    step = 0
    log_loss = 0
    for batch in dataloader:
        context_batch = batch["contexts"].to(
            "cuda", non_blocking=True
        )  # (batch_size, 4)
        center_batch = batch["centers"].to("cuda", non_blocking=True)  # (batch_size)

        optimizer.zero_grad()
        logits = embedding_layer(context_batch)
        loss = loss_fn(logits, center_batch)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 and step > 0:
            print(f"Step {step}, Avg loss (last {1000}): {log_loss / 1000:.4f}")
            log_loss = 0
        step += 1

    try:

        if os.path.exists("trained_model.pt"):
            torch.save(
                embedding_layer.state_dict(),
                f"trained_model_{D_EMBEDDINGS}_{CONTEXT_SIZE}.pt",
            )
        else:
            torch.save(
                embedding_layer.state_dict(),
                f"trained_model_{D_EMBEDDINGS}_{CONTEXT_SIZE}.pt",
            )

    except:
        torch.save(embedding_layer.state_dict(), f"trained_model_fallback.pt")

    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
