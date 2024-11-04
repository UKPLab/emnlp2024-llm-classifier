from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM

class Evaluater:
    def __init__(self) -> None:
       ''

    def merge_model(self, finetuned_model_dir:Path):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        model = AutoPeftModelForCausalLM.from_pretrained(
                    str(finetuned_model_dir)+'/',
                    torch_dtype=compute_dtype,
                    return_dict=False,
                    low_cpu_mem_usage=True,
                    device_map='cuda:0',
                )
        model = model.merge_and_unload()
        model.config.pad_token_id = tokenizer.pad_token_id

        if 'mistral' in str(finetuned_model_dir):
            model.config.sliding_window = 4096
       
        print('merge_model, 4model', model)
        
        return model, tokenizer

    def predict(self, test, model, tokenizer, labels, output_dir, response_key, is_val=False):
        if is_val:
            eval_file = output_dir / "VAL_eval_pred.csv"
        else:
            eval_file = output_dir / "eval_pred.csv"
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()
        
        for i in tqdm(range(len(test))):
            prompt = test[i]["text"]
            pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 10, 
                       )
            result = pipe(prompt)
            answer = result[0]['generated_text'].split(response_key)[-1]
            found = False
            for l in labels:
                if l.lower() in answer.lower():
                    pred = l
                    found = True
                    break
            if not found:
                pred="none"
            a = pd.DataFrame({ "true":[test[i]['label']], "pred":[pred], "answer":[answer], "prompt":[prompt]})
            a.to_csv(eval_file,mode="a",index=False,header=not eval_file.exists())
        return eval_file
        

    def evaluate(self, test, model=None, tokenizer=None, model_dir=None, output_dir=None,  do_predict = True, 
                 labels=None, label2id=None, id2label=None, 
                 emb_type=None, input_type=None, response_key = None, is_val = False):
        # load the model
        if model is None or tokenizer is None:
            model, tokenizer = self.merge_model(model_dir)

        start_time = pd.Timestamp.now()
        if output_dir is None:
                output_dir = Path(model_dir)
        if do_predict:
            eval_file = self.predict(test, model, tokenizer, labels, output_dir, response_key, is_val=is_val)
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time = inference_time.total_seconds()

        
        df = pd.read_csv(eval_file)
        none_nr = len(df[df['pred'] == 'none'])
        total_nr = len(df)

        eff = round((total_nr / int(inference_time)), 1)
        if is_val:
            inf_file = output_dir / "VAL_inference_time.json"
        else:
            inf_file = output_dir / "inference_time.json"
        with open (inf_file, 'w') as f:
            json.dump({'inference_time':int(inference_time), 'inference_efficieny':eff}, f, indent=4)


        #calculate accuarcy with 'none' samples
        y_pred = df["pred"]
        y_true = df["true"]
        # Map labels to ids
        label2id['none'] = len(label2id)
        map_func = lambda label: label2id[label]
        y_true = np.vectorize(map_func)(y_true)
        y_pred = np.vectorize(map_func)(y_pred)
        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        print(f'Accuracy: {accuracy:.3f}')
        
        # Generate accuracy report
        # if 'none' exists in the labels, add it to the target names for accuracy calculation
        if none_nr > 0:
            target_names = labels+['none']
        else:
            target_names = labels
        class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names, output_dict=True, zero_division=0)

        # but the marco avg and weighted avg f1 should not include 'none', otherwise 'none' class having 0 samples will affect the calculation
        if none_nr > 0:
            df = df[df['pred'] != 'none']
            y_pred = df["pred"]
            y_true = df["true"]
            map_func = lambda label: label2id[str(label)]
            y_true = np.vectorize(map_func)(y_true)
            y_pred = np.vectorize(map_func)(y_pred)
            class_report2 = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels, output_dict=True, zero_division=0)
            class_report['weighted avg 2'] = class_report2['weighted avg']
            class_report['macro avg 2'] = class_report2['macro avg']
            
        print('\nClassification Report:')
        class_report['none_nr'] = none_nr
        class_report['AIR'] = round(((total_nr - none_nr) / total_nr)*100, 1)
        print(class_report)

      
        if is_val:
            eval_file = output_dir / "VAL_eval_report.json"
        else:
            eval_file = output_dir / "eval_report.json"
        with open(str(eval_file), 'w') as f:
            json.dump(class_report, f, indent=4)

        if is_val:
            return accuracy
        
        
