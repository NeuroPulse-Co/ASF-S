"""
models.py - LeNet-300-100 for MNIST, plus helpers to load it and measure its size.

Architecture (paper Table 4: 98.51% test accuracy, 267K params, 0.53 MFLOPs):

    input 28x28 -> flatten 784
      -> fc1: Linear(784, 300) -> ReLU      <- the layer ASF-S prunes (paper Fig. 10)
      -> fc2: Linear(300, 100) -> ReLU
      -> fc3: Linear(100, 10)               <- output layer ("logits")

The class definition is restored from git commit 6b81a13
(CNN/cifar100_original_pipeline/models.py). Parameter names (fc1/fc2/fc3) must stay
identical so the paper's checkpoint loads with strict=True.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet300_100(nn.Module):
    def __init__(self, input_size=28 * 28, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x, return_acts=False):
        """
        return_acts=False -> logits only (normal use).
        return_acts=True  -> (logits, {"fc1": ..., "fc2": ..., "fc3": ...,
                                       "fc1_pre": ..., "fc2_pre": ...}).

        "fc1"/"fc2" are POST-ReLU activations; "fc1_pre"/"fc2_pre" are the values
        before ReLU, kept only so the "as_coded" variant can reproduce what the
        original scripts actually measured. The original extraction script
        (min_scripts/activation_ext3.py) put forward hooks on the nn.Linear modules,
        which fire BEFORE the functional F.relu, so it silently recorded pre-activation
        values (~80% negative for fc1). The paper's justification for cosine similarity
        (Sec. 3: "robust for non-negative activations (e.g., ReLU layers)") assumes
        post-ReLU values, so that is what we record. fc3 has no ReLU: its activations
        are the logits themselves.
        """
        x = torch.flatten(x, 1)
        z1 = self.fc1(x)          # pre-activation of fc1
        a1 = F.relu(z1)
        z2 = self.fc2(a1)         # pre-activation of fc2
        a2 = F.relu(z2)
        logits = self.fc3(a2)
        if return_acts:
            return logits, {"fc1": a1, "fc2": a2, "fc3": logits,
                            "fc1_pre": z1, "fc2_pre": z2}
        return logits


def load_lenet(path, device="cpu"):
    """Load a plain state_dict checkpoint. Tolerates layer widths other than 300/100,
    which lets us reload pruned checkpoints (fc1 with fewer than 300 neurons)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    n1, d_in = state["fc1.weight"].shape
    n2 = state["fc2.weight"].shape[0]
    n_cls = state["fc3.weight"].shape[0]
    model = LeNet300_100(input_size=d_in, num_classes=n_cls)
    # Resize fc1/fc2 if the checkpoint is a pruned model.
    model.fc1 = nn.Linear(d_in, n1)
    model.fc2 = nn.Linear(n1, n2)
    model.fc3 = nn.Linear(n2, n_cls)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def count_params(model):
    """Total parameter count, independent of requires_grad.

    The cifar100/ pipeline used ptflops, which skips frozen parameters; it therefore
    under-counted pruned models whose conv layers had been frozen (engineering-log #2).
    Counting numel() directly avoids that trap."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model):
    """FLOPs of one forward pass, counted analytically.

    A Linear(in, out) layer performs in*out multiply-accumulates (MACs); by the usual
    convention 1 MAC = 2 FLOPs. Bias adds and ReLUs are ignored (they are <0.2% here).
    For the unpruned model: 784*300 + 300*100 + 100*10 = 266,200 MACs = 0.53 MFLOPs,
    which matches the paper's Table 4."""
    macs = sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))
    return 2 * macs
