import collections
import sys
import torch

input_path = sys.argv[1]
output_path = sys.argv[2]

composer_model = torch.load(input_path)
print(composer_model["state"].keys())
print(composer_model["state"]["model"].keys())

hf_model = collections.OrderedDict()
for key in composer_model["state"]["model"]:
    print(key)
    print(key[6:])
    hf_model[key[6:]] = composer_model["state"]["model"][key]

torch.save(hf_model, output_path)
