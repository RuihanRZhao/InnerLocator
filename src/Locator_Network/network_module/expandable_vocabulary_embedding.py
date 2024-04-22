import torch
import torch.nn as nn


class ExpandableVocabularyEmbedding(nn.Module):
    def __init__(self, initial_vocab_size: int, initial_vocab_list: list, embedding_dim: int):
        super(ExpandableVocabularyEmbedding, self).__init__()
        self.embedding = nn.Embedding(initial_vocab_size, embedding_dim)
        self.vocab_size = initial_vocab_size
        self.embedding_dim = embedding_dim
        self.vocabulary_dict = {
            vocab: num for num, vocab in enumerate(initial_vocab_list)
        }

    def update_vocabulary(self, new_vocabulary):
        self.vocabulary_dict[new_vocabulary] = self.vocab_size

        new_embeddings = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim)
        new_embeddings.weight.data.normal_(0, 0.1)

        # match
        origin_embeddings_data = self.embedding.weight.data
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size + 1, embedding_dim=self.embedding_dim)
        self.embedding.weight.data[:self.vocab_size] = origin_embeddings_data
        self.embedding.weight.data[self.vocab_size:] = new_embeddings.weight.data

        self.vocab_size += 1

    def forward(self, x):
        vocab2index = []
        for item in x:
            if item not in self.vocabulary_dict:
                self.update_vocabulary(item)
            vocab2index.append(self.vocabulary_dict[item])

        return self.embedding(torch.tensor(vocab2index))


if __name__ == '__main__':
    test_embed = ExpandableVocabularyEmbedding(
        7,
        ["from", "from type", "to", "to type", "ToA", "AoA theta", "AoA phi"],
        3
    )
    test_embed.update_vocabulary("Central Router")

    print(test_embed(["Central Router", "from"]))
