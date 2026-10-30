"""Surrogate-gradient spiking neural network (LIF, BPTT) vs. an ANN of equal size on 8x8 digits.

Images are Poisson rate-coded into T time steps; a 2-layer leaky integrate-and-fire network is trained
end-to-end with backpropagation through time using a fast-sigmoid surrogate derivative of the spike
non-linearity. We report accuracy, spike sparsity and how accuracy depends on the number of time steps.
"""
import json, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)

class SpikeFn(torch.autograd.Function):
    scale = 10.0
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v); return (v > 0).float()
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (SpikeFn.scale * v.abs() + 1.0) ** 2          # fast-sigmoid surrogate

class SNN(nn.Module):
    def __init__(self, n_in=64, n_hid=128, n_out=10, beta=0.9, thr=1.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n_in, n_hid), nn.Linear(n_hid, n_out)
        self.beta, self.thr = beta, thr
    def forward(self, x, T):
        B = x.shape[0]; v1 = torch.zeros(B, self.fc1.out_features); v2 = torch.zeros(B, self.fc2.out_features)
        out = torch.zeros(B, 10); n_spk = 0.0
        for _ in range(T):
            s_in = (torch.rand_like(x) < x).float()                # Poisson rate coding
            v1 = self.beta * v1 + self.fc1(s_in)
            s1 = SpikeFn.apply(v1 - self.thr); v1 = v1 - s1 * self.thr   # soft reset
            v2 = self.beta * v2 + self.fc2(s1)
            out = out + v2 / T                                      # readout: mean membrane potential
            n_spk += s1.sum().item()
        return out, n_spk / (B * self.fc1.out_features * T)        # mean spikes / neuron / step

def data(seed=0):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    return [torch.tensor(a, dtype=torch.float32) for a in (Xtr, Xte)] + [torch.tensor(b) for b in (ytr, yte)]

def train(model, fwd, Xtr, ytr, epochs=60, lr=2e-3, seed=0):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]; loss = nn.functional.cross_entropy(fwd(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

if __name__ == "__main__":
    res = {"snn_acc": [], "ann_acc": [], "snn_spikes_per_neuron_step": []}
    for seed in range(3):
        Xtr, Xte, ytr, yte = data(seed); torch.manual_seed(seed)
        snn = SNN(); T = 25
        train(snn, lambda x: snn(x, T)[0], Xtr, ytr, seed=seed)
        with torch.no_grad():
            out, sp = snn(Xte, T); res["snn_acc"].append((out.argmax(1) == yte).float().mean().item())
            res["snn_spikes_per_neuron_step"].append(sp)
            if seed == 0:
                tsweep = {}
                for Tt in (2, 5, 10, 25, 50):
                    accs = [(snn(Xte, Tt)[0].argmax(1) == yte).float().mean().item() for _ in range(5)]
                    tsweep[Tt] = float(np.mean(accs))
        ann = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        train(ann, ann, Xtr, ytr, seed=seed)
        with torch.no_grad(): res["ann_acc"].append((ann(Xte).argmax(1) == yte).float().mean().item())
        print(seed, res["snn_acc"][-1], res["ann_acc"][-1], res["snn_spikes_per_neuron_step"][-1], flush=True)
    summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items()}
    summary["accuracy_vs_timesteps_seed0"] = tsweep
    json.dump(summary, open("results.json", "w"), indent=2); print(json.dumps(summary, indent=2))
    plt.figure(figsize=(4.5, 3.2)); plt.plot(list(tsweep), list(tsweep.values()), "o-")
    plt.xlabel("time steps T"); plt.ylabel("test accuracy"); plt.tight_layout(); plt.savefig("accuracy_vs_T.png", dpi=150)
