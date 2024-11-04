import shutil
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from .evaluater import Evaluater
                          

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
                  lora_r = 256,
                  lora_alpha = 256,
                  lora_dropout = 0.1,
                  learning_rate = 2e-4,
                  per_device_train_batch_size = 32,
                  train_epochs = 10,
                  output_dir = None,
                  bias = 'none',
                  target_modules="all-linear",
                  task_type = "CAUSAL_LM",
                  max_seq_length = 1024,
                  do_val = True,
                  response_key = '',
                  labels = None,
                  label2id = None):
        print('fine-tuning....')
        # Enable gradient checkpointing to reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()
        # Prepare the model for training 
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
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
                save_strategy="epoch",                    # save checkpoint every epoch
                gradient_checkpointing=True,              # use gradient checkpointing to save memory
                optim="paged_adamw_32bit",
                )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        
        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=max_seq_length,
                data_collator = data_collator,
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

        # Validate on the validation set and select the best checkpoint
        evaluater = Evaluater()
        print('Validating...')
        if do_val:
            best_acc = (None, None) 
            for d in output_dir.iterdir():
                if d.is_dir() and d.stem.startswith('checkpoint'):
                    print(d)
                    acc = evaluater.evaluate(val_ds, model_dir=d, labels=labels, label2id=label2id, response_key=response_key, is_val=True)
                    if best_acc == (None, None):
                        best_acc = (d.stem, acc)
                    else:
                        if acc > best_acc[1]:
                            best_acc = (d.stem, acc)
            #copy the best ckp to output_dir
            print('The best checkpoint is: ', best_acc[0], '..copying')
            for f in (output_dir/best_acc[0]).iterdir():
                if (output_dir/f.name).exists():
                    (output_dir/f.name).unlink()
                shutil.copy(f, output_dir/f.name)




        