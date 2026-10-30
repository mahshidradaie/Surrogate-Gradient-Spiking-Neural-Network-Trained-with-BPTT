# Surrogate-Gradient Spiking Neural Network vs. ANN (8×8 digits)

A 64-128-10 leaky integrate-and-fire network trained end-to-end with backpropagation through time. The
non-differentiable spike is replaced in the backward pass by a fast-sigmoid surrogate derivative. Inputs are
Poisson rate-coded over T time steps; the readout is the mean output membrane potential. Compared against a
ReLU MLP of identical size (64-128-10), 3 train/test splits (70/30), PyTorch.

```
pip install torch numpy scikit-learn matplotlib
python snn.py     # writes results.json and accuracy_vs_T.png
```

## Results (`results.json`)
| Model | Test accuracy (mean ± sd, 3 splits) |
|---|---|
| SNN (T=25) | 97.6 ± 0.2 % |
| ANN (ReLU MLP) | 97.0 ± 0.5 % |

Hidden layer fires on average 0.30 spikes/neuron/step (not very sparse). Accuracy vs. time steps (split 0):
T=2: 89.8 %, T=5: 95.6 %, T=10: 97.3 %, T=25: 97.4 %, T=50: 97.9 %.

## Caveats
Digits is an easy, small dataset, so SNN ≈ ANN says little about harder tasks; no energy measurement; no
tuning of β, threshold or surrogate slope beyond one default setting.
