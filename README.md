# Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions
This is the official code repository for the paper "Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions", presented at EMNLP 2024 main conference. It contains the scripts for the fine-tuning approaches outlined in the paper.

Please find the paper [here](https://arxiv.org/abs/2410.02028), and star the repository to stay updated with the latest information.

In case of questions please contact [Qian Ruan](mailto:ruan@ukp.tu-darmstadt.de).

## Abstract
Classification is a core NLP task architecture with many potential applications. While large language models (LLMs) have brought substantial advancements in text generation, their potential for enhancing classification tasks remains underexplored. To address this gap, we propose a framework for thoroughly investigating fine-tuning LLMs for classification, including both generation- and encoding-based approaches. We instantiate this framework in edit intent classification (EIC), a challenging and underexplored classification task. Our extensive experiments and systematic comparisons with various training approaches and a representative selection of LLMs yield new insights into their application for EIC. We investigate the generalizability of these findings on five further classification tasks. To demonstrate the proposed methods and address the data shortage for empirical edit analysis, we use our bestperforming EIC model to create Re3-Sci2.0, a new large-scale dataset of 1,780 scientific document revisions with over 94k labeled edits. The quality of the dataset is assessed through human evaluation. The new dataset enables an in-depth empirical study of human editing behavior in academic writing. 

## Approaches
![](/resource/approaches.png)

*Figure 2. Proposed approaches with a systematic investigation of the key components: input types (red), language models (green), and transformation functions (yellow). See §3 and §4 of the paper for details.*

## Quickstart
1. Download the project from github.
```bash
git clone https://github.com/UKPLab/llm_classifier
```

2. Setup environment
```bash
python -m venv .llm_classifier
source ./.llm_classifier/bin/activate
pip install -r requirements.txt
```   
   

### Fine-tuining LLMs
Check the 'finetune_EIC_\<X\>.py' scripts for the complete workflows with each approach: Gen, SeqC, XNet and SNet. You can customize the arguments within \<settings\> and \</settings\>. Refer to the paper for more details.

For example, fine-tune LLM with the SeqC approach:

1. Basic Settings

```python
    # basic settings
    # <settings>
    task_name ='edit_intent_classification'
    method = 'finetuning_llm_seqc' # select an approach from ['finetuning_llm_gen','finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']
    train_type ='train' # name of the training data in data/Re3-Sci/tasks/edit_intent_classification
    val_type = 'val' # name of the validation data in data/Re3-Sci/tasks/edit_intent_classification
    test_type = 'test' # name of the test data in data/Re3-Sci/tasks/edit_intent_classification
    # </settings>
```
2. Load Data

```python
    from tasks.task_data_loader import TaskDataLoader
    task_data_loader = TaskDataLoader(task_name=task_name, train_type=train_type, val_type=val_type, test_type=test_type)
    train_ds, val_ds, test_ds= task_data_loader.load_data()
    labels, label2id, id2label = task_data_loader.get_labels()
```

3. Load Model

```python
    # load model from path
    # <settings>
    model_path = 'path/to/model'
    emb_type = None # transformation function for xnet and snet approaches, select from [''diff', diffABS', 'n-diffABS', 'n-o', 'n-diffABS-o'], None for SeqC and Gen
    input_type='text_st_on'  #input type for the model, select from ['text_nl_on', 'text_st_on', 'inst_text_st_on', 'inst_text_nl_on'] for natural language input, structured input, instruction + structured input,  instruction + natural language input, respectively
    # </settings>
    from tasks.task_model_loader import TaskModelLoader
    model_loader = TaskModelLoader(task_name=task_name, method=method).model_loader
    model, tokenizer = model_loader.load_model_from_path(model_path, labels=labels, label2id=label2id, id2label=id2label, emb_type=emb_type, input_type=input_type)
```
4. Preprocess Data

```python
    # <settings>
    max_length = 1024
    # </settings>
    from tasks.task_data_preprocessor import TaskDataPreprocessor
    data_preprocessor = TaskDataPreprocessor(task_name=task_name, method=method).data_preprocessor
    train_ds = data_preprocessor.preprocess_data(train_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
    val_ds = data_preprocessor.preprocess_data(val_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
    test_ds = data_preprocessor.preprocess_data(test_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
```
5. Fine-tune Model

```python
    # fine-tune model
    # <settings>
    lora_r = 128 # LoRA rank parameter
    lora_alpha = 128 # Alpha parameter for LoRA scaling
    lora_dropout = 0.1 # Dropout probability for LoRA layers
    learning_rate = 2e-4 # Learning rate
    per_device_train_batch_size = 32 # Batch size per GPU for training 
    train_epochs = 10 # Number of epochs to train
    recreate_dir = True # Create a directory for the model
    # </settings>
    # create model dir to save the fine-tuned model
    from finetune_EIC_SeqC import create_model_dir
    output_dir = create_model_dir(task_name, method, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, emb_type, input_type, recreate_dir=recreate_dir)
    # fine-tune
    from tasks.task_model_finetuner import TaskModelFinetuner
    model_finetuner = TaskModelFinetuner(task_name=task_name, method=method).model_finetuner
    model_finetuner.fine_tune(model, tokenizer, train_ds = train_ds , val_ds = val_ds,  lora_r = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout,
                                   learning_rate = learning_rate, per_device_train_batch_size = per_device_train_batch_size, train_epochs = train_epochs, output_dir = output_dir)
```
6. Evaluate

```python
    # fine-tune model
    # evaluate the fine-tuned model
    from tasks.task_evaluater import TaskEvaluater
    evaluater = TaskEvaluater(task_name=task_name, method=method).evaluater
    evaluater.evaluate(test_ds, model_dir=output_dir, labels=labels, label2id=label2id, id2label=id2label, emb_type=emb_type, input_type=input_type, response_key=response_key)
```

## Citation

Please use the following citation:

```
@inproceedings{ruan2024llmclassifier,
      title={Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions}, 
      author={Qian Ruan and Ilia Kuznetsov and Iryna Gurevych},
      year={2024},
      eprint={2410.02028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02028}, 
}
```

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

<https://intertext.ukp-lab.de/>

<https://www.ukp.tu-darmstadt.de>

<https://www.tu-darmstadt.de>
