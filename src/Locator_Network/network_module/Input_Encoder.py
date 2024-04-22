import torch
import torch.nn as nn

from expandable_vocabulary_embedding import ExpandableVocabularyEmbedding


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
        output = torch.stack(output, dim=0).squeeze(2)

        output_mask = ~torch.isnan(output)
        output[torch.isnan(output)] = 0

        return output, output_mask


if __name__ == '__main__':
    test_Encoder = Input_Encoder(0, [], 1)
    encoder_output = test_Encoder([[['from', 'Central Router'], ['from type', 'Router'], ['to', 'Router01'], ['to type', 'Router'],
                         ['ToA', 3.4729082487272894e-08], ['AoA theta', -1.5869858934760817],
                         ['AoA phi', -1.5869858934760817]],
                        [['from', 'Router03'], ['from type', 'Router'], ['to', 'Tag005'], ['to type', 'Target'],
                         ['ToA', 5.8878809665227115e-08], ['AoA theta', -1.0621641127827242],
                         ['AoA phi', -1.0621641127827242]],
                        [['from', 'Router03'], ['from type', 'Router'], ['to', 'Phone008'], ['to type', 'Endpoint'],
                         ['ToA', 2.5351112393236074e-08], ['AoA theta', 1.2563360327930881],
                         ['AoA phi', 1.2563360327930881]],
                        [['from', 'Tag000'], ['from type', 'Target'], ['to', 'Tag014'], ['to type', 'Target'],
                         ['ToA', 2.8544444049627134e-08], ['AoA theta', None], ['AoA phi', None]],
                        [['from', 'Central Router'], ['from type', 'Router'], ['to', 'Tag001'], ['to type', 'Target'],
                         ['ToA', 1.99519205254038e-08], ['AoA theta', -2.109167511524454],
                         ['AoA phi', -2.109167511524454]],
                        [['from', 'Tag001'], ['from type', 'Target'], ['to', 'Tag002'], ['to type', 'Target'],
                         ['ToA', 4.986046294361071e-08], ['AoA theta', None], ['AoA phi', None]],
                        [['from', 'Tag001'], ['from type', 'Target'], ['to', 'Tag003'], ['to type', 'Target'],
                         ['ToA', 2.9628972150723896e-08], ['AoA theta', None], ['AoA phi', None]],
                        ]
                       )

    print(
        encoder_output[0].shape, encoder_output[1].shape
          )
