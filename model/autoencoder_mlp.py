"""
Backward-compatible wrapper — main implementation migrated to model.autoencoder_pinn.

This file exists so that old code importing AutoencoderPINN from model.autoencoder_mlp
continues to work.  New code should import directly from model.autoencoder_pinn.
"""
from .autoencoder_pinn import (
    ModulatedResidualBlock,
    AmplitudeNet,
    SharedPINNEncoder,
    TaylorComplexDecoder,
    AutoencoderTeukolskyPINN,
)

AutoencoderPINN = AutoencoderTeukolskyPINN
