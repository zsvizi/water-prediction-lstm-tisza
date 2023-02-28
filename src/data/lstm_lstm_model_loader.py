import os
import yaml

from src import PROJECT_PATH
from src.model.lstm_lstm_model import LSTMSequenceModel
from src.data.lstm_lstm_model_data_preprocessor import LSTMLSTMModelDataPreprocessor


class LSTMLSTMModelLoader:
    def __init__(self):
        self.p_yaml_dict = None
        self.model = None
        self.hyperparameters = None

    def __load_hyperparameters(self):
        self.hyperparameters = {
            "selected_gauges": self.p_yaml_dict['time_varying_reals_encoder'],
            "features": self.p_yaml_dict['time_varying_reals_encoder'],
            "max_encoder_length": self.p_yaml_dict['max_encoder_length'],
            "max_prediction_length": self.p_yaml_dict['max_prediction_length'],
            "scalers": self.p_yaml_dict['scalers'],
            "normalizer": self.p_yaml_dict['output_transformer'],
            "target": self.p_yaml_dict['target']
        }

    def load_model(self, model_name: str):
        LSTMLSTMModelDataPreprocessor.get_model_files(model_name=model_name)
        self.p_yaml_dict = yaml.load(open(os.path.join(PROJECT_PATH, "data", "hparams_" + model_name + ".yaml")),
                                     Loader=yaml.Loader)
        self.model = LSTMSequenceModel(**self.p_yaml_dict)
        self.__load_hyperparameters()
