import torch
from embeddings import EmbeddingLayer
from transformers import AutoTokenizer
from torchtext.vocab import GloVe


def find_nearest_neighbors(embeddings: torch.Tensor, k=100):
    similarity_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1))
    _, indices = torch.topk(similarity_matrix, k=k + 1, dim=1)
    return indices[:, 1:]


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    index_to_token = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    model = EmbeddingLayer(50257, 4, 20)
    print("before -> ", model.token_embedding_table.weight.data)
    model.load_state_dict(torch.load("trained_model.pt"))
    print("after -> ", model.token_embedding_table.weight.data)

    embeddings = model.token_embedding_table.weight.data
    nearest_neighbors = find_nearest_neighbors(embeddings)
    word_to_lookup = 26239
    to_check = nearest_neighbors[word_to_lookup]
    print("Checking neighbors for:", index_to_token[word_to_lookup])
    for index in to_check:
        print(index_to_token[index])
    # 23132


if __name__ == "__main__":
    main()
