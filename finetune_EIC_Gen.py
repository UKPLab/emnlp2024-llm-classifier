from pathlib import Path
import shutil

def create_model_dir(task_name, method, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, emb_type, input_type, recreate_dir=True):
    # create model dir
    output_dir = Path("./results")
    if not output_dir.exists():
            output_dir.mkdir()
    output_dir = output_dir/task_name
    if not output_dir.exists():
            output_dir.mkdir()
    output_dir = output_dir/method
    if not output_dir.exists():
            output_dir.mkdir()
    model_folder_name = Path(model_path).stem +'_'+ f'_lora-r{lora_r}-a{lora_alpha}-d{lora_dropout}_lr{learning_rate}'
    model_folder_name += f'_bs{per_device_train_batch_size}_ep{train_epochs}_{train_type}_{test_type}_ml{max_length}_{emb_type}_{input_type}'
    output_dir = Path(output_dir/model_folder_name)
    if output_dir.exists():
         if recreate_dir:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
    else:
        output_dir.mkdir()
    return output_dir
        

def main():
    # basic settings
    # <settings>
    task_name ='edit_intent_classification'
    method = 'finetuning_llm_gen' # select an approach from ['finetuning_llm_gen','finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']
    train_type ='train' # name of the training data in data/Re3-Sci/tasks/edit_intent_classification
    val_type = 'val' # name of the validation data in data/Re3-Sci/tasks/edit_intent_classification
    test_type = 'test' # name of the test data in data/Re3-Sci/tasks/edit_intent_classification
    # </settings>
    print('========== Basic settings: ==========')
    print(f'task_name: {task_name}')
    print(f'method: {method}')
    print(f'train_type: {train_type}')
    print(f'val_type: {val_type}')
    print(f'test_type: {test_type}')
    ############################################################################
    # load task data
    from tasks.task_data_loader import TaskDataLoader
    task_data_loader = TaskDataLoader(task_name=task_name, train_type=train_type, val_type=val_type, test_type=test_type)
    train_ds, val_ds, test_ds= task_data_loader.load_data()
    labels, label2id, id2label = task_data_loader.get_labels()
    print('========== 1. Task data loaded: ==========')
    print(f'train_ds: {train_ds}')
    print(f'val_ds: {val_ds}')
    print(f'test_ds: {test_ds}')
    print(f'labels: {labels}')
    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')
    ############################################################################
    # load model from path
    # <settings>
    model_path = 'path/to/model'
    emb_type = None # transformation function for xnet and snet approaches, select from [''diff', diffABS', 'n-diffABS', 'n-o', 'n-diffABS-o'], None for SeqC and Gen
    #input type for the model, select from ['text_nl_on', 'text_st_on', 'inst_text_st_on', 'inst_text_nl_on'] 
    #for natural language input, structured input, instruction + structured input,  instruction + natural language input, respectively
    input_type='inst_text_st_on' 
    # </settings>
    
    from tasks.task_model_loader import TaskModelLoader
    model_loader = TaskModelLoader(task_name=task_name, method=method).model_loader
    model, tokenizer = model_loader.load_model_from_path(model_path, labels=labels, label2id=label2id, id2label=id2label, emb_type=emb_type, input_type=input_type)
    print('========== 2. Model loaded: ==========')
    print(f'model: {model}')
    

    ############################################################################
    # preprocess dataset
    # <settings>
    max_length = 1024
    # </settings>
    from tasks.task_data_preprocessor import TaskDataPreprocessor
    data_preprocessor = TaskDataPreprocessor(task_name=task_name, method=method).data_preprocessor
    if method in ['finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']:
        train_ds = data_preprocessor.preprocess_data(train_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        val_ds = data_preprocessor.preprocess_data(val_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        test_ds = data_preprocessor.preprocess_data(test_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        response_key = None
    elif method in ['finetuning_llm_gen']:
        train_ds,_ = data_preprocessor.preprocess_data(train_ds, max_length=max_length, input_type=input_type, is_train=True)
        val_ds,_ = data_preprocessor.preprocess_data(val_ds, max_length=max_length, input_type=input_type, is_train=False)
        test_ds, response_key = data_preprocessor.preprocess_data(test_ds, max_length=max_length, input_type=input_type, is_train=False)
    
    print('========== 3. Dataset preprocessed: ==========')
    print('train_ds: ', train_ds[0])
    print('val_ds: ', val_ds[0])
    print('test_ds: ', test_ds[0])
    print('response_key: ', response_key)
    
    ############################################################################
    # fine-tune model
    # <settings>
    lora_r = 256 # LoRA rank parameter
    lora_alpha = 256 # Alpha parameter for LoRA scaling
    lora_dropout = 0.1 # Dropout probability for LoRA layers
    learning_rate = 2e-4 # Learning rate
    per_device_train_batch_size = 32 # Batch size per GPU for training 
    train_epochs = 10 # Number of epochs to train
    recreate_dir = True # Create a directory for the model, if true, the existing directory will be removed and recreated
    # </settings>
    # create model dir to save the fine-tuned model
    output_dir = create_model_dir(task_name, method, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, emb_type, input_type, recreate_dir=recreate_dir)
    print('========== 4. Model dir created: ==========')
    print('output_dir: ', output_dir)
    # fine-tune
    from tasks.task_model_finetuner import TaskModelFinetuner
    model_finetuner = TaskModelFinetuner(task_name=task_name, method=method).model_finetuner
    if method in ['finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']:
         model_finetuner.fine_tune(model, tokenizer, train_ds = train_ds , val_ds = val_ds,  lora_r = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout,
                                   learning_rate = learning_rate, per_device_train_batch_size = per_device_train_batch_size, train_epochs = train_epochs, output_dir = output_dir)
    elif method in ['finetuning_llm_gen']:
         model_finetuner.fine_tune(model, tokenizer, train_ds = train_ds , val_ds = val_ds,  lora_r = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout,
                                   learning_rate = learning_rate, per_device_train_batch_size = per_device_train_batch_size, train_epochs = train_epochs, output_dir = output_dir, 
                                   do_val = True, response_key = response_key, labels = labels, label2id=label2id)
    print('========== 5. Model fine-tuned: ==========')
    print('output_dir: ', output_dir)
    

    ############################################################################
    # evaluate fine-tuned model
    from tasks.task_evaluater import TaskEvaluater
    evaluater = TaskEvaluater(task_name=task_name, method=method).evaluater
    evaluater.evaluate(test_ds, model_dir=output_dir, labels=labels, label2id=label2id, id2label=id2label, emb_type=emb_type, input_type=input_type, response_key=response_key)
    print('========== 6. Model evaluated ==========')
    print('output_dir: ', output_dir)
    print('========== DONE ==========')
   

if __name__ == "__main__":
    main()
