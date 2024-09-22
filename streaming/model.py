import torch
import torch.nn as nn
from streaming.encoder_decoder import StreamingEncoder, StreamingDecoder
from streamvc.model import StreamVC
from streamvc.f0 import F0Estimator
from streamvc.energy import EnergyEstimator
from streamvc._utils import auto_batching


class StreamingStreamVC(nn.Module):
    def __init__(self, model: StreamVC, target_speech: torch.Tensor):
        super().__init__()

        with torch.no_grad():
            target_latent = model.speech_pooling(
                model.speech_encoder(target_speech))
        
        self.target_latent: torch.Tensor = target_latent
        self.content_encoder: StreamingEncoder = StreamingEncoder(model.content_encoder)
        self.f0_estimator: F0Estimator = model.f0_estimator
        self.energy_estimator: EnergyEstimator = model.energy_estimator
        self.decoder: StreamingDecoder = StreamingDecoder(model.decoder)

    def forward(self, source_speech_chunck: torch.Tensor):
        with torch.no_grad():
            target_latent = self.target_latent
            content_latent = self.content_encoder(source_speech_chunck)

            f0 = self.f0_estimator(source_speech_chunck)
            energy = self.energy_estimator(
                source_speech_chunck).unsqueeze(dim=-1)
            
            source_linguistic_features = torch.cat(
                [content_latent, f0, energy], dim=-1)
            
            return self.decoder(source_linguistic_features, self.target_latent)
