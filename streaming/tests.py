"""Test that all objects used in the StreamingStreamVC model are torchscript compatible"""

import torch
import torch.nn as nn
import unittest
from typing import List

from streaming.model import StreamingStreamVC
from streaming.encoder_decoder import StreamingEncoder, StreamingDecoder, StreamingEncoderBlock, StreamingDecoderBlock, StreamingResidualUnit
from streaming.modules import StreamingCausalConv1d
from streamvc.model import StreamVC
from streamvc.encoder_decoder import Encoder, Decoder, EncoderBlock, DecoderBlock, ResidualUnit
from streamvc.modules import CausalConv1d, CausalConvTranspose1d, FiLM
from streamvc.energy import EnergyEstimator
from streamvc.f0 import F0Estimator

class JitTest(unittest.TestCase):
    def test_streaming_conv1d(self):
        base = CausalConv1d(10, 10, 3)
        test_scripable(StreamingCausalConv1d(base))

    def test_streaming_conv_transpose1d(self):
        test_scripable(CausalConvTranspose1d(10, 10, 3))
    
    def test_film(self):
        test_scripable(FiLM(10, 10))
    
    def test_streaming_residual_unit(self):
        base = ResidualUnit(10, 10, 10)
        test_scripable(StreamingResidualUnit(base))
    
    def test_streaming_encoder_block(self):
        base = EncoderBlock(10, 10, 10)
        test_scripable(StreamingEncoderBlock(base))
    
    def test_streaming_decoder_block(self):
        base = DecoderBlock(10, 10, 10)
        test_scripable(StreamingDecoderBlock(base))
    
    def test_streaming_encoder(self):
        base = Encoder(10, 10)
        test_scripable(StreamingEncoder(base))

    def test_streaming_decoder(self):
        base = Decoder(10, 10, 10)
        test_scripable(StreamingDecoder(base))
    
    def test_energy_estimator(self):
        test_scripable(EnergyEstimator())

    def test_f0_estimator(self):
        test_scripable(F0Estimator())
    
    def test_streaming_model(self):
        base = StreamVC()
        test_scripable(StreamingStreamVC(base, torch.rand(2, 10000)))

# TODO: Equivalences tests between streaming and non-streaming modules
    

def test_scripable(model: nn.Module):
    jitted = torch.jit.script(model)
    return True



if __name__ == '__main__':

    unittest.main()