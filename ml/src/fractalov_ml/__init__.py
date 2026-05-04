"""Fractalov ML pipeline.

Stage 5: dataset generation by calling the Java backend's /render endpoint with
randomised recipes. The on-disk artefacts (PNG images + a parquet of labels +
a parquet of stratified splits) are designed to be consumed directly by the
PyTorch Dataset class in :mod:`fractalov_ml.dataset` for Stage 6 training.
"""

__version__ = "0.1.0"
