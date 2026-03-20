#!/usr/bin/env python3
“”“Add SmearGate + BigramHash to train_gpt.py. Run after cloning your fork.”””
import re
text = open(‘train_gpt.py’).read()

# 1. Add env vars

old = ’    use_swiglu = bool(int(os.environ.get(“USE_SWIGLU”, “0”)))’
new = ‘’’    use_swiglu = bool(int(os.environ.get(“USE_SWIGLU”, “0”)))
use_smeargate = bool(int(os.environ.get(“USE_SMEARGATE”, “0”)))
bigram_hash_buckets = int(os.environ.get(“BIGRAM_HASH_BUCKETS”, 0))
bigram_hash_dim = int(os.environ.get(“BIGRAM_HASH_DIM”, 128))
muon_weight_decay = float(os.environ.get(“MUON_WEIGHT_DECAY”, 0.0))’’’
text = text.replace(old, new)

# 2. Add SmearGate + BigramHash classes before Block class

old_block = ‘class Block(nn.Module):’
new_block = ‘’’class SmearGate(nn.Module):
def **init**(self, dim):
super().**init**()
self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
def forward(self, x):
g = torch.sigmoid(self.gate.to(dtype=x.dtype))
prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
return (1 - g) * x + g * prev

class BigramHash(nn.Module):
def **init**(self, vocab_size, num_buckets, hash_dim, model_dim):
super().**init**()
self.num_buckets = num_buckets
self.hash_emb = nn.Embedding(num_buckets, hash_dim)
self.proj = CastedLinear(hash_dim, model_dim, bias=False)
self.proj._zero_init = True
def forward(self, input_ids):
prev = F.pad(input_ids[:, :-1], (1, 0))
pair_hash = (prev.long() * 1000003 + input_ids.long()) % self.num_buckets
return self.proj(self.hash_emb(pair_hash))

class Block(nn.Module):’’’
text = text.replace(old_block, new_block)

# 3. Add to GPT.**init** signature

old_init = ‘use_swiglu=False,’
new_init = ‘use_swiglu=False, use_smeargate=False, bigram_hash_buckets=0, bigram_hash_dim=128,’

# Only replace in GPT class definition (first occurrence after class GPT)

text = text.replace(old_init, new_init, 1)

# 4. Add SmearGate and BigramHash to GPT.**init** body after tok_emb

old_emb = ’        self.tok_emb = nn.Embedding(vocab_size, model_dim)’
new_emb = ‘’’        self.tok_emb = nn.Embedding(vocab_size, model_dim)
self.smeargate = SmearGate(model_dim) if use_smeargate else None
self.bigram_hash = BigramHash(vocab_size, bigram_hash_buckets, bigram_hash_dim, model_dim) if bigram_hash_buckets > 0 else None’’’
text = text.replace(old_emb, new_emb, 1)

# 5. Apply in forward pass

old_fwd = ‘’’    def forward(self, input_ids, target_ids’’’

# Find the forward method and add SmearGate + BigramHash after tok_emb

old_fwd_body = ’        x = self.tok_emb(input_ids)\n’
new_fwd_body = ‘’’        x = self.tok_emb(input_ids)
if self.smeargate is not None:
x = self.smeargate(x)
if self.bigram_hash is not None:
x = x + self.bigram_hash(input_ids)
‘’’
text = text.replace(old_fwd_body, new_fwd_body, 1)

# 6. Pass new args to GPT constructor

old_construct = ‘use_swiglu=args.use_swiglu,’
new_construct = ‘use_swiglu=args.use_swiglu, use_smeargate=args.use_smeargate, bigram_hash_buckets=args.bigram_hash_buckets, bigram_hash_dim=args.bigram_hash_dim,’
text = text.replace(old_construct, new_construct, 1)

# 7. Add Muon weight decay

old_muon_step = ‘’’            curr = 0
for p in params:
g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
p.add_(g, alpha=-lr)
curr += p.numel()’’’
new_muon_step = ‘’’            wd = group.get(“weight_decay”, 0.0)
curr = 0
for p in params:
if wd > 0:
p.mul_(1 - wd * lr)
g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
p.add_(g, alpha=-lr)
curr += p.numel()’’’
text = text.replace(old_muon_step, new_muon_step, 1)

# 8. Add weight_decay to Muon constructor

old_muon_init = ‘def **init**(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):’
new_muon_init = ‘def **init**(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):’
text = text.replace(old_muon_init, new_muon_init, 1)

old_muon_super = ‘super().**init**(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))’
new_muon_super = ‘super().**init**(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))’
text = text.replace(old_muon_super, new_muon_super, 1)

# 9. Pass weight_decay when creating Muon

old_muon_create = ‘Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)’
new_muon_create = ‘Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)’
text = text.replace(old_muon_create, new_muon_create, 1)

open(‘train_gpt.py’, ‘w’).write(text)
print(“Patched: SmearGate + BigramHash + MuonWD added!”)
print(“Enable: USE_SMEARGATE=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 MUON_WEIGHT_DECAY=0.01”)
