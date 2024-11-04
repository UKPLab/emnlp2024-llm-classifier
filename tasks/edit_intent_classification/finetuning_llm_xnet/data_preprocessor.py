class DataPreprocessor:
    def __init__(self) -> None:
        '''
        '''
        
        print('Preprocessing the data ...XNet')

    def preprocess_data(self, dataset, label2id, tokenizer, max_length=1024, input_type='text_st_on'): 
        """
        :param dataset: Hugging Face dataset
        :param label2id (dict): label to id mapping
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens for padding and truncation
        :param input_type (str): type of input text 
        """
        # perpare input text and label
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_type = input_type
        print("input_type: ", input_type)
        print('max_length: ', max_length)
        dataset = dataset.map(self.create_input_text_and_label, keep_in_memory=True)
        # Shuffle dataset
        seed = 42
        dataset = dataset.shuffle(seed = seed)
        return dataset
    
    def create_input_text_and_label(self, sample):
        """
        Creates a formatted input text for a sample
        :param sample: sample from the dataset
        """
        text_src = sample['text_src'] if sample['text_src'] is not None else ''
        text_tgt = sample['text_tgt'] if sample['text_tgt'] is not None else ''

        if self.input_type == 'text_st_on':
            sample["text_old"] = f"<old> {text_tgt} </old>"
            sample["text_new"] = f"<new> {text_src} </new>"
            sample["text"] = f"<old> {text_tgt} </old>" + '\n ' + f"<new> {text_src} </new>"
        elif self.input_type == 'text_nl_on':
            sample["text_old"] = text_tgt
            sample["text_new"] = text_src
            sample["text"] = text_tgt + '\n ' + text_src
        else:
            raise ValueError("Invalid input type. Choose from ['text_st_on','text_nl_on']")
        
        emb1 = self.tokenizer.encode_plus(sample["text_old"],return_tensors="pt")["input_ids"]
        emb2 = self.tokenizer.encode_plus(sample["text_new"],return_tensors="pt")["input_ids"]
        sample['emb1'] = emb1
        sample['emb2'] = emb2

        sample['input_ids_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]

        label = self.label2id[sample['label']]
        sample["label"] = label
      
        return sample
