import collections
import sys
import torch

input_path = sys.argv[1]
output_path = sys.argv[2]

composer_model = torch.load(input_path)

hf_model = collections.OrderedDict()
for key in composer_model["state"]["model"]:
    smaller_tensor = composer_model["state"]["model"][key]
    hf_model[key] = smaller_tensor

torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(hf_model, prefix="model.")

torch.save(hf_model, output_path)
