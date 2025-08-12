# Simulated Annealing Quantizer

This directory now contains a reference implementation of a simulated
annealing (SA) based weight quantizer integrated into ``quant_modules.py``.

## Features

* Searches for the best 6‑bit quantization levels using multi‑dimensional
  simulated annealing to minimise mean squared error (MSE).
* Detects outliers with the ``|x| > mean + 3 × std`` rule.  Outliers are
  encoded with 12 bits while a neighbouring position is sacrificed to store
  the flag so that the tensor remains aligned.
* Helper functions to quantize and dequantize tensors.

## Usage

```python
import torch
from antquant.quant_modules import annealing_quantize, dequantize

weights = torch.randn(1024)
result = annealing_quantize(weights)
reconstructed = dequantize(result)
```

`result` contains the quantization levels (basis), 6‑bit codes, outlier mask
and 12‑bit encoded outlier values.  The `dequantize` helper reconstructs the
original tensor including outliers.

The module is self‑contained and can be integrated into the existing training
pipeline or used for further research on quantization strategies.
