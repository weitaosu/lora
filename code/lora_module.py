import torch
import torch.nn as nn


class LoRAConv1D(nn.Module):
    """Wrap a GPT-2 Conv1D and add LoRA to the requested attention slices.
    For c_attn: targets is a subset of {'q','k','v'} indexing the packed Q/K/V output.
    For c_proj (output projection): targets == ['o'] and the delta is added to the full output.
    """
    def __init__(self, base, rank, alpha, dropout, hidden_dim, targets):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.hidden_dim = hidden_dim
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.targets = list(targets)
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self.targets:
            A = nn.Parameter(torch.zeros(rank, hidden_dim))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = nn.Parameter(torch.zeros(hidden_dim, rank))   # B init zero -> initial delta is zero
            self.lora_A[name] = A
            self.lora_B[name] = B

    def forward(self, x):
        out = self.base(x)
        x_d = self.drop(x)
        if self.targets == ['o']:
            delta = (x_d @ self.lora_A['o'].T) @ self.lora_B['o'].T * self.scaling
            return out + delta
        chunks = list(torch.split(out, self.hidden_dim, dim=-1))
        idx = {'q': 0, 'k': 1, 'v': 2}
        for name in self.targets:
            delta = (x_d @ self.lora_A[name].T) @ self.lora_B[name].T * self.scaling
            chunks[idx[name]] = chunks[idx[name]] + delta
        return torch.cat(chunks, dim=-1)
    
def inject_lora(model, rank, alpha, dropout, hidden_dim, targets):
    qkv = [t for t in targets if t in ('q', 'k', 'v')]
    o   = 'o' in targets
    for block in model.transformer.h:
        if qkv:
            block.attn.c_attn = LoRAConv1D(
                block.attn.c_attn, rank, alpha,
                dropout, hidden_dim, qkv,
            )
        if o:
            block.attn.c_proj = LoRAConv1D(
                block.attn.c_proj, rank, alpha,
                dropout, hidden_dim, ['o'],
            )
    for n, p in model.named_parameters():
        p.requires_grad = ('lora_' in n)
