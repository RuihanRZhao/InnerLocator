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
        self.tensor_to_index_dict = {}
        self.reverse_vocabulary_dict = {num: vocab for vocab, num in self.vocabulary_dict.items()}

    def update_vocabulary(self, new_vocabulary):
        self.vocabulary_dict[new_vocabulary] = self.vocab_size

        new_embeddings = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim)
        new_embeddings.weight.data.normal_(0, 0.1)

        # match
        origin_embeddings_data = self.embedding.weight.data
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size + 1, embedding_dim=self.embedding_dim)
        self.embedding.weight.data[:self.vocab_size] = origin_embeddings_data
        self.embedding.weight.data[self.vocab_size:] = new_embeddings.weight.data

        # Update reverse dictionary
        self.reverse_vocabulary_dict[self.vocab_size] = new_vocabulary

        self.vocab_size += 1

    def forward(self, x):
        vocab2index = []
        for item in x:
            if item not in self.vocabulary_dict:
                self.update_vocabulary(item)
            vocab2index.append(self.vocabulary_dict[item])

        output = self.embedding(torch.tensor(vocab2index))

        for num, item in enumerate(vocab2index):
            self.tensor_to_index_dict[output.squeeze(1)[num]] = item

        return output

    def tensor_to_words(self, target):
        # Convert a list of indices back to their corresponding words

        return self.reverse_vocabulary_dict[self.tensor_to_index_dict[target]] if target in self.tensor_to_index_dict else ""


class Input_Encoder(nn.Module):
    def __init__(self, initial_vocab_size, initial_vocab_list, embedding_dim):
        super(Input_Encoder, self).__init__()
        self.text_embedding = ExpandableVocabularyEmbedding(initial_vocab_size, initial_vocab_list, embedding_dim)

    def forward(self, input_list: list):
        text_data = []
        numeric_data = []
        for row in input_list:
            text_row = []
            numeric_row = []
            for pack in row:
                if isinstance(pack[1], str):
                    text_row.append(pack)
                else:
                    numeric_row.append([pack[0], (pack[1] if pack[1] is not None else float('nan'))])
            text_data.append(text_row)
            numeric_data.append(numeric_row)

        # normalize ToA data
        data_to_normalize = torch.tensor([i[0][1] for i in numeric_data])

        normalized_data = (data_to_normalize - torch.mean(data_to_normalize))/torch.std(data_to_normalize)

        for num, i in enumerate(numeric_data):
            i[0] = ['ToA', normalized_data[num].item()]

        numeric_tensors = []
        for record in numeric_data:
            numeric_tensors.append(torch.tensor([item[1] for item in record]))

        text_tensors = []
        for record in text_data:
            values = [item[1] for item in record]
            text_tensors.append(self.text_embedding(values))

        output = [torch.cat((text_tensors[i], numeric_tensors[i].unsqueeze(1)), dim=0) for i in range(len(text_tensors))]
        output = torch.stack(output, dim=0)

        output_mask = ~torch.isnan(output)
        output[torch.isnan(output)] = 0

        return output, output_mask

    def decode_output(self, target):
        vocab_output = []

        for item in target:
            vocab_output.append(self.text_embedding.tensor_to_words(item))

        return vocab_output


if __name__ == '__main__':
    test_embed = ExpandableVocabularyEmbedding(
        7,
        ["from", "from type", "to", "to type", "ToA", "AoA theta", "AoA phi"],
        3
    )
    test_embed.update_vocabulary("Central Router")

    print(test_embed(["Central Router", "from"]))

    # test_Encoder = Input_Encoder(0, [], 1)
    # encoder_output = test_Encoder([[['from', 'Central Router'], ['from type', 'Router'], ['to', 'Router01'], ['to type', 'Router'],
    #                      ['ToA', 3.4729082487272894e-08], ['AoA theta', -1.5869858934760817],
    #                      ['AoA phi', -1.5869858934760817]],
    #                     [['from', 'Router03'], ['from type', 'Router'], ['to', 'Tag005'], ['to type', 'Target'],
    #                      ['ToA', 5.8878809665227115e-08], ['AoA theta', -1.0621641127827242],
    #                      ['AoA phi', -1.0621641127827242]],
    #                     [['from', 'Router03'], ['from type', 'Router'], ['to', 'Phone008'], ['to type', 'Endpoint'],
    #                      ['ToA', 2.5351112393236074e-08], ['AoA theta', 1.2563360327930881],
    #                      ['AoA phi', 1.2563360327930881]],
    #                     [['from', 'Tag000'], ['from type', 'Target'], ['to', 'Tag014'], ['to type', 'Target'],
    #                      ['ToA', 2.8544444049627134e-08], ['AoA theta', None], ['AoA phi', None]],
    #                     [['from', 'Central Router'], ['from type', 'Router'], ['to', 'Tag001'], ['to type', 'Target'],
    #                      ['ToA', 1.99519205254038e-08], ['AoA theta', -2.109167511524454],
    #                      ['AoA phi', -2.109167511524454]],
    #                     [['from', 'Tag001'], ['from type', 'Target'], ['to', 'Tag002'], ['to type', 'Target'],
    #                      ['ToA', 4.986046294361071e-08], ['AoA theta', None], ['AoA phi', None]],
    #                     [['from', 'Tag001'], ['from type', 'Target'], ['to', 'Tag003'], ['to type', 'Target'],
    #                      ['ToA', 2.9628972150723896e-08], ['AoA theta', None], ['AoA phi', None]],
    #                     ]
    #                    )
    #
    # print(
    #     encoder_output[0].shape, encoder_output[1].shape
    #       )
