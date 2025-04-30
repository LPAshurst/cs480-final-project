import torch
from embedding_model import EmbeddingLayer
from torch.nn import functional as F
from transformers import AutoTokenizer
from torchtext.vocab import GloVe


def find_nearest_neighbors(embeddings: torch.Tensor, index: int, k=100):

    query = embeddings[index].unsqueeze(0)
    similarity_matrix = F.cosine_similarity(query, embeddings)
    _, indices = torch.topk(similarity_matrix, k=k + 1)
    return indices[1:]


def main():
    g = GloVe(name="6B", dim=100)
    word_to_index = g.stoi
    index_to_word = g.itos
    word_to_index["dog"]
    t = g.vectors
    helper = find_nearest_neighbors(g.vectors, word_to_index["dog"])

    set_to_check_glove = set([index_to_word[index.item()] for index in helper])
    print(set_to_check_glove)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    index_to_token = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    token_to_index = tokenizer.get_vocab()

    model = EmbeddingLayer(50257, 64)
    model.load_state_dict(torch.load("trained_model_64_4_100000.pt"))

    embeddings = model.token_embedding_table.weight.data
    nearest_neighbors = find_nearest_neighbors(embeddings, token_to_index["dog"])

    set_to_check_custom = set(
        [index_to_token[index.item()] for index in nearest_neighbors]
    )
    print(set_to_check_custom)
    print(len(set_to_check_glove.intersection(set_to_check_custom)))
    # 23132


if __name__ == "__main__":
    main()
