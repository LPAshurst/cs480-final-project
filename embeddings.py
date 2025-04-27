import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import sys

print("starting...", file=sys.stderr)

# hyperparameters
BATCH_SIZE = 64  # how many independent sequences will we process in parallel?
CONTEXT_SIZE = 4  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
(
    torch.set_default_device("cuda")
    if torch.cuda.is_available()
    else torch.set_default_device("cpu")
)
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
        input_token_size,
        d_model,
    ):
        super().__init__()

        # define some important vars
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.input_size = input_token_size

        self.token_embedding_table = torch.nn.Embedding(
            vocab_size,
            d_model,
        )
        self.linear_one = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embeds: torch.Tensor = self.token_embedding_table(x)
        input_embeds = embeds.mean(dim=0, keepdim=True)

        out: torch.Tensor = self.linear_one(input_embeds)
        return out.squeeze(0)


def process_input(tokenizer, num_proc):

    ds = load_dataset("upstage/Pretraining_Dataset", split="train")

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=False,
        )

    tokenized_ds = ds.map(tokenize_batch, batched=True, num_proc=num_proc)
    flat_encoded = [token for example in tokenized_ds["input_ids"] for token in example]
    return torch.tensor(flat_encoded, dtype=torch.long)


def build_cbow_pairs(data, context_size=2):
    for i in range(context_size, len(data) - context_size):
        left = data[i - context_size : i]
        right = data[i + 1 : i + context_size + 1]
        context = torch.cat((left, right))
        center = data[i]
        yield context, center


def main():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_proc = os.cpu_count() / 2

    data = process_input(tokenizer, num_proc)

    d_model = 20
    vocab_size = tokenizer.vocab_size
    embedding_layer = EmbeddingLayer(vocab_size, vocab_size, d_model)
    embedding_layer.to("cuda")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=1e-3)

    # train the model
    for epoch in range(1):
        step = 0
        log_loss = 0
        for context, target in build_cbow_pairs(data, CONTEXT_SIZE):

            context = context.to("cuda")
            target = target.to("cuda")

            optimizer.zero_grad()
            logits = embedding_layer(context)

            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            log_loss += loss.item()

            if step % 1000 == 0 and step > 0:
                print(f"Step {step}, Avg loss (last {1000}): {log_loss / 1000:.4f}")
                log_loss = 0
            step += 1

            # save the model
            if os.path.exist("trained_model.pt"):
                torch.save(embedding_layer.state_dict(), "trained_model2.pt")
            torch.save(embedding_layer.state_dict(), "trained_model.pt")
            print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
