from pathlib import Path
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

class ModelLoader:
    def __init__(self) -> None:
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
            )
        
    def load_model_from_path(self, model_path:str, device_map='auto', 
                             labels=None, label2id=None, id2label=None, 
                             emb_type=None, input_type=None):

        print('Loading model from...', model_path)	

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    quantization_config = self.bnb_config,
                                                    device_map = device_map, 
                                                    ) 
        if 'Mistral' in model_path:
            model.config.sliding_window = 4096
        return model, tokenizer
    

