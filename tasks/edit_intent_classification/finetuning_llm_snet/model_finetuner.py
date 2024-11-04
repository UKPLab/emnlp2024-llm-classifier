import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments   
import evaluate
accuracy = evaluate.load("accuracy")


def collate_fn(examples, device=None):
    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_tuple'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_tuple'])
        example['label'] = torch.as_tensor(example['label'])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    if device is not None:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
    return {"input_ids": input_ids, 
            "attention_mask": attention_masks,
            "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class ModelFinetuner:
    def __init__(self) -> None:
        ''

    def print_trainable_parameters(self, model, use_4bit = False):
        """Prints the number of trainable parameters in the model.
        :param model: PEFT model
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2
        print(
            f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
        )
    
    def fine_tune(self, 
                  model,
                  tokenizer,
                  train_ds = None,
                  val_ds = None,
                  lora_r = 128,
                  lora_alpha = 128,
                  lora_dropout = 0.1,
                  learning_rate = 2e-4,
                  per_device_train_batch_size = 32,
                  train_epochs = 10,
                  output_dir = None,
                  bias = 'none',
                  target_modules="all-linear",
                  task_type = None,
                  max_seq_length = 4096
                  ):
        print('fine-tuning....')
        # Prepare the model for training 
        model = prepare_model_for_kbit_training(model)
       
        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
            modules_to_save = ['score']
        )
        model = get_peft_model(model, peft_config)
        # Print information about the percentage of trainable parameters
        self.print_trainable_parameters(model)
       
        args = TrainingArguments(
                output_dir = output_dir,
                num_train_epochs=train_epochs,
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = 8,
                learning_rate = learning_rate, 
                logging_steps=10,
                fp16 = True,
                weight_decay=0.001,
                max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                max_steps=-1,
                warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                group_by_length=True,
                lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                report_to="tensorboard",                  # report metrics to tensorboard
                evaluation_strategy="epoch",              # save checkpoint every epoch
                save_strategy="epoch",
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="paged_adamw_32bit",
                remove_unused_columns=False,
                load_best_model_at_end=True, 
                metric_for_best_model="eval_accuracy",
                label_names = ['labels'],
                 )
       
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=max_seq_length,
                data_collator = collate_fn,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                } 
            )

        model.config.use_cache = False
        do_train = True

        # Launch training and log metrics
        print("Training...")

        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
