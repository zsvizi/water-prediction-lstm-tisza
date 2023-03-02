from abc import ABC
from typing import Dict, List

from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM
import torch


class LSTMSequenceModel(AutoRegressiveBaseModelWithCovariates, ABC):
    """Custom model with one target - uses teacher forcing"""

    def __init__(
            self,
            target: str,
            target_lags: Dict[str, Dict[str, int]],
            n_layers: int,
            n_features: int,
            hidden_size: int,
            time_varying_reals_encoder: List[str],
            dropout: float,
            is_target_a_feature: bool,
            target_pos: int,
            is_teacher_forcing: bool,
            scalers,
            max_prediction_length: int = 5,
            max_encoder_length: int = 30,
            static_categoricals=None,
            time_varying_categoricals_encoder=None,
            time_varying_categoricals_decoder=None,
            static_reals=None,
            time_varying_reals_decoder=None,
            x_reals=None,
            x_categoricals=None,
            embedding_labels=None,
            embedding_paddings=None,
            categorical_groups=None,
            embedding_sizes=None,
            **kwargs,
    ):
        if embedding_sizes is None:
            embedding_sizes = []
        if categorical_groups is None:
            categorical_groups = []
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = []
        if x_categoricals is None:
            x_categoricals = []
        if x_reals is None:
            x_reals = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if static_categoricals is None:
            static_categoricals = []
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # ENCODER LSTM - number of inputs is n_features (it might contain the target)
        self.lstm_encode = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=self.hparams.n_features,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        # DECODER LSTM - number of inputs is hidden_size passed from encoder
        self.lstm_decode = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=1,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        # OUTPUT TRANSFORMERS
        self.encoder_transformer = torch.nn.Linear(self.hparams.hidden_size, 1)
        self.output_layer = torch.nn.Linear(self.hparams.hidden_size, 1)

    def encode(self,
               x: Dict[str, torch.Tensor]):
        # SELECT INPUT [batch_size, encoder_length, n_features]
        input_vector = x["encoder_cont"].clone()

        # ENCODE output [batch_size, encoder_length, hidden_size]
        # hidden_state tupple(2) -> 2x [num_layers, batch_size, hidden_size]
        encoder_output, encoder_hidden = self.lstm_encode(input_vector,
                                                          lengths=x["encoder_lengths"],
                                                          enforce_sorted=False)

        # LAST ENCODER OUTPUT TO DECODER transformed [batch_size, 1, 1]
        decoder_input = self.encoder_transformer(encoder_output[:, -1:, :])

        return decoder_input, encoder_hidden

    def decode(self,
               x: Dict[str, torch.Tensor],
               encoder_output: torch.Tensor,
               hidden_state: torch.Tensor):
        # SELECT FIRST INPUT
        if self.hparams.is_target_a_feature:
            # LAST ENCODER TARGET CAN BE USED [batch_size, 1, 1]
            # the same would be used in the case of teacher forcing
            # x["encoder_target"] [batch_size, encoder_length]
            input_vector = x["encoder_cont"][:, -1:, self.hparams.target_pos].unsqueeze(2)
        else:
            # target cannot be used, encoder output is used instead [batch_size, 1, 1]
            input_vector = encoder_output

        # INITIALIZE OUTPUTS [decoder_lengths, batch_size, 1, 1]
        target_len = x["decoder_lengths"][0].item()
        batch_size = x["decoder_lengths"].size()[0]
        outputs = torch.zeros(target_len, batch_size, 1, 1, device=self.device)

        # PREDICT RECURSIVELY
        for t in range(target_len):
            # DECODER output [batch_size, 1, hidden_size]
            decoder_output, hidden_state = self.lstm_decode(input_vector,
                                                            hidden_state,
                                                            enforce_sorted=False)
            # TRANSFORM TO OUTPUT FORMAT [batch_size, 1, 1]
            decoder_output = self.output_layer(decoder_output)
            # COLLECT DECODER OUTPUTS
            outputs[t] = decoder_output

            # SELECT NEXT INPUT FOR THE NEXT STEP OF DECODER
            if self.training and self.hparams.is_teacher_forcing:
                # LAST DECODER TARGET CAN BE USED
                input_vector = x["decoder_cont"][:, t:t + 1, self.hparams.target_pos].unsqueeze(2)
            else:  # prediction or no teacher forcing
                # USE DECODER OUTPUT AS THE NEXT INPUT VECTOR
                input_vector = decoder_output

        # RESHAPE OUTPUTS TO FORCASTING FORMAT
        # outputs [decoder_lengths, batch_size, 1, 1] -> [batch_size, decoder_lengths, 1]
        outputs = torch.swapaxes(outputs, 0, 1)
        prediction = torch.squeeze(outputs, 3)

        # rescale outputs
        prediction = self.transform_output(prediction,
                                           target_scale=x["target_scale"])

        return prediction

    def forward(self,
                x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ENCODE
        decoder_input, encoder_hidden = self.encode(x)  # encode to hidden state

        # DECODE
        prediction = self.decode(x, decoder_input, encoder_hidden)

        # FORCASTING NETWORK OUTPUT
        return self.to_network_output(prediction=prediction)
