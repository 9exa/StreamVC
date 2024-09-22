import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from typing import List

from streaming.modules import StreamingCausalConv1d
from streamvc.encoder_decoder import Encoder, Decoder, EncoderBlock, DecoderBlock, ResidualUnit
from streamvc.modules import FiLM


class StreamingEncoder(nn.Module):
    def __init__(self, base: Encoder):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make 
        # the same checks that are made in the original class
        encoder_blocks = [
            StreamingEncoderBlock(block) for block in base.encoder[3:7]
        ]
        conv1 = StreamingCausalConv1d(base.encoder[1])
        conv7 = StreamingCausalConv1d(base.encoder[7])

        self.encoder = nn.Sequential(
            Rearrange('... samples -> ... 1 samples'),
            conv1,
            nn.ELU(),
            *encoder_blocks,
            conv7,
            nn.ELU(),
            Rearrange('... embedding frames -> ... frames embedding')
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class StreamingDecoder(nn.Module):
    def __init__(self, base: Decoder):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make 
        # the same checks that are made in the original class

        # Need to manually interleave FiLM layers with the decoder blocks
        # because nn.Sequential does not support conditional execution
        blocks = [
            StreamingDecoderBlock(block) for block in base.decoder_blocks.blocks
        ]
        films = base.decoder_blocks.films

        self.decoder_blocks = DecoderBlocks(blocks, films)
        self.preblocks = nn.Sequential(
            Rearrange('... frames embedding -> ... embedding frames'),
            StreamingCausalConv1d(base.preblocks[1]),
            nn.ELU()
        )

        self.postblocks = nn.Sequential(
            StreamingCausalConv1d(base.postblocks[0]),
            Rearrange('... 1 samples -> ... samples')
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        x = self.preblocks(x)
        x = self.decoder_blocks(x, condition)
        return self.postblocks(x)

class StreamingEncoderBlock(nn.Module):
    def __init__(self, base: EncoderBlock):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make 
        # the same checks that are made in the original class

        self.block = nn.Sequential(
            StreamingResidualUnit(base.block[0]),
            StreamingResidualUnit(base.block[1]),
            StreamingResidualUnit(base.block[2]),
            StreamingCausalConv1d(base.block[3]),
            nn.ELU()
        )


    def forward(self, x: torch.Tensor):
        return self.block(x)


class StreamingDecoderBlock(nn.Sequential):
    def __init__(self, base: DecoderBlock):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make 
        # the same checks that are made in the original class

        self.block = nn.Sequential(
            base.block[0],
            StreamingResidualUnit(base.block[1]),
            StreamingResidualUnit(base.block[2]),
            StreamingResidualUnit(base.block[3]),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)

class DecoderBlocks(nn.Module):
    def __init__(self, blocks: List[StreamingDecoderBlock], films: List[FiLM]):
        assert len(blocks) == len(films)
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.films = nn.ModuleList(films)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        for block, film in zip(self.blocks, self.films):
            x = film(block(x), condition)
        return x


class StreamingResidualUnit(nn.Module):
    def __init__(self, base: ResidualUnit):
        super().__init__()

        # NOTE: If you want to implement this calss in C++, you should make 
        # the same checks that are made in the original class

        self.unit = nn.Sequential(
            StreamingCausalConv1d(base.unit[0]),
            nn.ELU(),
            StreamingCausalConv1d(base.unit[2]),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor):
        return self.unit(x) + x
