class DataPreprocessor:
    def __init__(self) -> None:
        print('Preprocessing the data...SeqC')

    def preprocess_data(self, dataset, label2id, tokenizer, max_length=1024, input_type='text_st_on'):
        """
        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param input_type (str): Type of input text
        """
        # perpare input text and label
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_type = input_type
        print("input_type: ", input_type)
        print('max_length: ', max_length)
        dataset = dataset.map(lambda x: self.create_input_text_and_label(x, self.input_type),keep_in_memory=True)
        # Shuffle dataset
        seed = 42
        dataset = dataset.shuffle(seed = seed)
        return dataset
    
    def create_input_text_and_label(self, sample, input_type):
        """
        Creates a formatted input text for a sample
        :param sample: sample from the dataset
        """
        instruction = "Classify the intent of the following sentence edit. The possible labels are: Grammar, Clarity, Fact/Evidence, Claim, Other. " 
        text_src = sample['text_src'] if sample['text_src'] is not None else ''
        text_tgt = sample['text_tgt'] if sample['text_tgt'] is not None else ''

        if input_type == 'text_st_on':
            sample["text"] = f"<old> {text_tgt} </old>" + '\n ' + f"<new> {text_src} </new>"
        elif input_type == 'text_nl_on':
            sample["text"] = text_tgt + '\n ' + text_src
        elif input_type == 'inst_text_st_on':
            sample["text"] = instruction + '\n ' + f"<old> {text_tgt} </old>" + '\n ' + f"<new> {text_src} </new>"
        elif input_type == 'inst_text_nl_on':
            sample["text"] = instruction + '\n ' + text_tgt + '\n ' + text_src
        else:
            raise ValueError("Invalid input type. Choose from ['text_st_on','text_nl_on','inst_text_st_on', 'inst_text_st_on']")

        label = self.label2id[sample['label']]
        sample["label"] = label

        sample['input_ids_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        sample['attention_mask_text'] = self.tokenizer.encode_plus(sample["text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
        return sample
    