import torch
from embedding_model import EmbeddingLayer
from torch.nn import functional as F
from transformers import AutoTokenizer
from torchtext.vocab import GloVe

MODEL_FILE_NAME = "trained_model_64_4_500000.pt"


def create_word_to_token(
    words: list, glove_word_to_index: dict, gpt2_word_to_index: dict
):
    word_to_indexes = {}
    for word in words:
        glove_index = glove_word_to_index[word]
        custom_index = gpt2_word_to_index[word]
        word_to_indexes[word] = (glove_index, custom_index)
    return word_to_indexes


def find_nearest_neighbors(embeddings: torch.Tensor, index: int, k=100):
    query = embeddings[index].unsqueeze(0)
    similarity_matrix = F.cosine_similarity(query, embeddings)
    _, indices = torch.topk(similarity_matrix, k=k + 1)
    return indices[1:]


def run_tests(
    word_to_token_both, index_to_word, index_to_token, embeddings, glove_vectors
):
    with open("tests-output.txt", "a") as f:
        for word, (glove_index, custom_index) in word_to_token_both.items():

            intersection_count = -1
            k = 0
            while intersection_count <= 0:
                k += 10
                glove_nearest_neighbors = find_nearest_neighbors(
                    glove_vectors, glove_index, k=k
                )
                set_to_check_glove = set(
                    [index_to_word[index.item()] for index in glove_nearest_neighbors]
                )

                custom_nearest_neighbors = find_nearest_neighbors(
                    embeddings, custom_index, k=k
                )
                custom_to_check_glove = set(
                    [index_to_token[index.item()] for index in custom_nearest_neighbors]
                )
                set_inter = set_to_check_glove.intersection(custom_to_check_glove)
                intersection_count = len(set_inter)

            f.write(
                f"There was an intersection of {intersection_count} for the word: {word} with a k value of {k}\n"
            )
            f.write(f"The intersected words are as follows: {str(set_inter)}\n\n")


def main():

    words = [
        "dog",
        "moon",
        "king",
        "bag",
        "usa",
        "man",
        "jump",
        "food",
        "terror",
        "bytes",
    ]

    # initialize GloVe
    g = GloVe(name="6B", dim=100)
    glove_word_to_index = g.stoi
    index_to_word = g.itos

    # initialize gpt2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    index_to_token = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    gpt2_word_to_index = tokenizer.get_vocab()

    # get embedddings
    model = EmbeddingLayer(50257, 64)
    model.load_state_dict(torch.load(MODEL_FILE_NAME))
    embeddings = model.token_embedding_table.weight.data

    #  get a tuple of the matching idx for the corresponding tokens
    word_to_token_both: dict = create_word_to_token(
        words, glove_word_to_index, gpt2_word_to_index
    )

    parsed_model_name = MODEL_FILE_NAME.split("_")
    with open("tests-output.txt", "w") as f:
        f.write(
            f"Showing results for a model trained on {parsed_model_name[-1][:-3]} rows with an embedding dim of {parsed_model_name[-3]}, a context size of {parsed_model_name[-2]}\n\n"
        )
    # run the tests
    run_tests(word_to_token_both, index_to_word, index_to_token, embeddings, g.vectors)


if __name__ == "__main__":
    main()
