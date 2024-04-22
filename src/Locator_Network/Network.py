import torch
import torch.nn as nn

from network_module.Input_Encoder import Input_Encoder


class InnerLocatorNetwork(nn.Module):
    def __init__(
            self,
            initial_vocab_size,
            initial_vocab_list,
            vocab_embedding_dim,
            feature_dim,
            num_layers_Transformer,
            num_heads_Transformer,
            hidden_dim_Transformer,
            hidden_dim_LSTM,
            num_layers_LSTM
    ):
        super(InnerLocatorNetwork, self).__init__()
        self.encoder = Input_Encoder(
            initial_vocab_size,
            initial_vocab_list,
            vocab_embedding_dim
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads_Transformer,
            dim_feedforward=hidden_dim_Transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers_Transformer
        )
        self.logic_network = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim_LSTM,
            num_layers=num_layers_LSTM
        )

        # Transformer Decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads_Transformer,
            dim_feedforward=hidden_dim_Transformer
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=num_layers_Transformer
        )

        self.output_control = nn.Conv1d(in_channels=7, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, input_data, output_name, output_target_num):
        encoded_data, encoded_mask = self.encoder(input_data)

        # Adjust input dimensions to [seq_len, batch, feature]
        encoded_data = encoded_data.permute(1, 0, 2)

        # Reshape x to collapse last dimension into features
        encoded_data = torch.flatten(encoded_data, start_dim=2)  # Now  is [7, x]

        # Transformer expects [seq_len, batch, features]
        transformer_encoded_logit = self.transformer_encoder(encoded_data)

        # LSTM expects [batch, seq_len, features], rearrange x to fit LSTM
        transformer_encoded_logit = transformer_encoded_logit.permute(1, 0, 2)  # Now  is [x, 7, feature_dim]

        # Pass through LSTM
        lstm_out, _ = self.logic_network(transformer_encoded_logit)
        transformer_decoded_output = self.transformer_decoder(lstm_out, transformer_encoded_logit)
        transformer_decoded_output = transformer_decoded_output.permute(1, 0, 2)

        output = self.output_control(transformer_decoded_output.squeeze(2)).permute(1, 0)

        if output.shape[0] >= output_target_num:
            output = output[:output_target_num, :]
        elif output.shape[0] < output_target_num:
            # If the output is shorter than required, we can pad it (simple zero-padding shown here)
            padding = torch.zeros((output_target_num - output.shape[0], 4), device=output.device)

            output = torch.cat([output, padding], dim=0)

        name_list = self.encoder.text_embedding(output_name).squeeze(1).tolist()
        return output, name_list


if __name__ == "__main__":
    initial_vocab_size = 0
    initial_vocab_list = []
    vocab_embedding_dim = 1
    feature_dim = 1
    num_layers_Transformer = 2
    num_heads_Transformer = 1
    hidden_dim_Transformer = 32
    hidden_dim_LSTM = 1
    num_layers_LSTM = 1

    brain = InnerLocatorNetwork(
        initial_vocab_size,
        initial_vocab_list,
        vocab_embedding_dim,
        feature_dim,
        num_layers_Transformer,
        num_heads_Transformer,
        hidden_dim_Transformer,
        hidden_dim_LSTM,
        num_layers_LSTM
    )
    result = brain(
        [
            [['from', 'Central Router'], ['from type', 'Router'], ['to', 'Router01'], ['to type', 'Router'],
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
        ], ["Tag001","Tag000"]
    )
    print(result)
