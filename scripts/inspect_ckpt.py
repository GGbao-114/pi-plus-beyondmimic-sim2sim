import sys
import torch

p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
print("type:", type(d))
if isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, dict):
            keys = list(v.keys())
            print(k, "dict", keys[:8])
        else:
            print(k, type(v), getattr(v, "shape", None))
