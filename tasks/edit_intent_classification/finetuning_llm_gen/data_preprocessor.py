# Initialize static strings for the prompt template
# natural language input (nl)
INSTRUCTION_KEY = "### Instruction:" 
INSTRUCTION_KEY_END = ''
INPUT_KEY = "INPUT:" 
INPUT_KEY_END = ''
NEW_START = 'NEW:'
NEW_END = ''
OLD_START = 'OLD:'
OLD_END = ''
RESPONSE_KEY = 'RESPONSE:'
END_KEY = '### End'

 
#structured input (st)
INSTRUCTION_KEY_ST = "<instruction>" 
INSTRUCTION_KEY_END_ST = '</instruction>'
INPUT_KEY_ST = '<input>'
INPUT_KEY_END_ST = '</input>'
NEW_START_ST = '<new>'
NEW_END_ST = '</new>'
OLD_START_ST = '<old>'
OLD_END_ST = '</old>'
RESPONSE_KEY_ST = "<response>" 
END_KEY_ST = "</response>" 


TASK_PROMPT = "Classify the intent of the following sentence edit. The possible labels are: Grammar, Clarity, Fact/Evidence, Claim, Other. " 

PROMPT_ST_DIC = {'nl': [INSTRUCTION_KEY,INSTRUCTION_KEY_END, INPUT_KEY, INPUT_KEY_END, OLD_START,OLD_END, NEW_START, NEW_END, RESPONSE_KEY, END_KEY],
                 'st': [INSTRUCTION_KEY_ST,INSTRUCTION_KEY_END_ST, INPUT_KEY_ST, INPUT_KEY_END_ST, OLD_START_ST,OLD_END_ST, NEW_START_ST, NEW_END_ST, RESPONSE_KEY_ST, END_KEY_ST]}

class DataPreprocessor:
    def __init__(self) -> None:
       print('Preprocessing the data...Gen')

    def preprocess_data(self, dataset, max_length=1024, input_type='text_st_on', is_train:bool=True):
        """
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param input_type (str): Type of input text
        """
        self.prompt_st_type = input_type.split('_')[-2]
        instruction_key, instruction_key_end, input_key, input_key_end, old_start, old_end, new_start, new_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        
        # Add prompt to each sample
        print("Preprocessing dataset...")
        if is_train:
            dataset = dataset.map(self.create_prompt_formats_train, keep_in_memory=True)
        else:
            dataset = dataset.map(self.create_prompt_formats_test, keep_in_memory=True)
       
        # Shuffle dataset
        seed = 42
        dataset = dataset.shuffle(seed = seed)
        return dataset, response_key
    
    def create_prompt_formats_train(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction_key, instruction_key_end, input_key, input_key_end, old_start, old_end, new_start, new_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        task_prompt = TASK_PROMPT
        # Combine a prompt with the static strings
        instruction = f"{instruction_key} {task_prompt} {instruction_key_end}"

        text_src = sample['text_src'] if sample['text_src'] is not None else ''
        text_tgt = sample['text_tgt'] if sample['text_tgt'] is not None else ''
        input_context = f"{input_key}\n {old_start} {text_tgt} {old_end}\n {new_start} {text_src} {new_end}\n{input_key_end}" 
        response = f"{response_key}{sample['label']}"
        end = f"{end_key}"
        # Create a list of prompt template elements
        parts = [part for part in [instruction, input_context, response, end] if part]
        # Join prompt template elements into a single string to create the prompt template
        formatted_prompt = "\n".join(parts)
        # Store the formatted prompt template in a new key "text"
        sample["text"] = formatted_prompt
        return sample
    
    def create_prompt_formats_test(self, sample):
        """
        Creates a formatted prompt template for a prompt in the dataset
        :param sample: sample from the dataset
        """
        instruction_key, instruction_key_end, input_key, input_key_end, old_start, old_end, new_start, new_end, response_key, end_key = PROMPT_ST_DIC[self.prompt_st_type]
        task_prompt = TASK_PROMPT
        instruction = f"{instruction_key} {task_prompt} {instruction_key_end}"
        text_src = sample['text_src'] if sample['text_src'] is not None else ''
        text_tgt = sample['text_tgt'] if sample['text_tgt'] is not None else ''
        input_context = f"{input_key}\n {old_start} {text_tgt} {old_end}\n {new_start} {text_src} {new_end}\n{input_key_end}" 
        response = f"{response_key}"
        parts = [part for part in [instruction, input_context, response] if part]
        formatted_prompt = "\n".join(parts)
        sample["text"] = formatted_prompt
        return sample
        