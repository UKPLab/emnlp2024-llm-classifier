from pathlib import Path
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
                         

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

        model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                 quantization_config = self.bnb_config,
                                                 device_map = device_map, 
                                                 num_labels = len(labels),
                                                 ) 
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.id2label = id2label
        model.config.label2id = label2id
        if 'Mistral' in model_path:
            model.config.sliding_window = 4096
        return model, tokenizer
    
