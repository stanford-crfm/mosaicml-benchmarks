import torch
from composer.utils import reproducibility
from transformers import DataCollatorForLanguageModeling
from transformers.models.gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from llm.hf_flash_gpt2 import GPT2FlashLMHeadModel


def test_fwd_bkw(config_path, autocast_device, autocast_dtype):
    reproducibility.seed_all(42)

    # Build both models
    shared_config = GPT2Config.from_json_file(config_path)
    non_flash_model = GPT2LMHeadModel(shared_config)
    flash_model = GPT2FlashLMHeadModel(shared_config)

    # Initialize with same parameters
    non_flash_state_dict = non_flash_model.state_dict()
    flash_model.load_state_dict(non_flash_state_dict)

    # Fake inputs
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    fake_sample = tokenizer('Here is a fake sample of length 8')
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    fake_batch = collate_fn([fake_sample])


    # Move to device
    non_flash_model = non_flash_model.to(autocast_device)
    flash_model = flash_model.to(autocast_device)
    fake_batch = {
        k: v.to(autocast_device)
        for k, v in fake_batch.items()
    }
    print (fake_batch)

    # Compare outputs
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        non_flash_outputs = non_flash_model(**fake_batch).logits
        flash_outputs = flash_model(**fake_batch).logits

    print ('#'*20)
    print ('OUTPUTS')
    print (non_flash_outputs)
    print (flash_outputs)
    print (torch.allclose(flash_outputs, non_flash_outputs, atol=5e-02))



config_path = './hf_configs/tests/gpt-125m-ctx-1024-no-dropout.json'

autocast_device = 'cuda'
autocast_dtype = torch.bfloat16
test_fwd_bkw(config_path, autocast_device, autocast_dtype)
