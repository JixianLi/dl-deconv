from model.deep_cnn import DeepCNN
from model.espcn import ESPCN


def build_model(model_config):
    if model_config.name == "espcn":
        return ESPCN(channels=model_config.channels)
    if model_config.name == "deep_cnn":
        return DeepCNN(num_layers=model_config.num_layers, channels=model_config.channels)
    raise ValueError(f"unknown model: {model_config.name}")
