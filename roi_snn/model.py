"""The shallow ROI SNN: 7x7 conv -> LIF -> 3x3 conv -> LIF -> ROI spike map.

Each frame is one timestep; temporal context lives only in the LIF membranes,
which are carried across frames (no extra latency).

beta values of the form 1 - 2**-k map to a shift-only leak in hardware
(mem -= mem >> k), so the sweep defaults use those.
"""
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

SHIFT_BETAS = (0.5, 0.75, 0.875, 0.9375, 0.96875)  # k = 1..5


class RoiSNN(nn.Module):
    def __init__(self, in_ch=1, hidden=8, beta=0.875, threshold=1.0, learn_beta=False):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv2d(in_ch, hidden, 7, padding=3)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg, learn_beta=learn_beta)
        self.conv2 = nn.Conv2d(hidden, 1, 3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg, learn_beta=learn_beta)
        self.threshold = threshold

    def forward(self, x):
        """x: (T, B, C, H, W). Returns dict of per-step tensors stacked on T.

        logits = output membrane - threshold (pre-reset), used for the loss;
        spk2 is the actual ROI map the hardware would emit.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, spk2_rec, logit_rec = [], [], []
        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.conv1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            logit_rec.append(mem2 - self.threshold)
        return dict(spk1=torch.stack(spk1_rec), spk2=torch.stack(spk2_rec),
                    logits=torch.stack(logit_rec))
