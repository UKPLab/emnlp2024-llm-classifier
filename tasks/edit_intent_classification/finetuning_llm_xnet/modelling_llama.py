# Built upon the huggingface implementation 

from typing import List, Optional, Tuple, Union
from transformers import LlamaModel, LlamaPreTrainedModel
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaForSequenceClassificationCross(LlamaPreTrainedModel):
    def __init__(self, config, model=None, score=None, emb_type=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.emb_type = emb_type
       
        if model is None:
            self.model = LlamaModel(config)
        else:
            self.model = model

        if score is None:
            if self.emb_type in['diff','diffABS']:
                input_size = config.hidden_size
            elif self.emb_type in ['n-o','n-diffABS']:
                input_size = config.hidden_size*2
            elif self.emb_type in ['n-diffABS-o']:
                input_size = config.hidden_size*3
            else:
                raise ValueError("invalid emb_type")
            self.score = nn.Linear(input_size, self.num_labels, bias=False)
            
        else:
            self.score = score
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def length_to_mask(self, lens, max_len, device):
        lens = lens.to(dtype=torch.long)
        lens = lens.to(device)
        base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
        base = base.to(device)
        return base < lens.unsqueeze(1)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        len_old: Optional[torch.LongTensor] = None,
        len_new: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths0 = -1
            sequence_lengths1 = -1
        else:
            if input_ids is not None:
                # the first pad_token_id is the end of the 2nd sentence
                sequence_lengths1 = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1) # new (2nd) sentence last token
                sequence_lengths0 = len_old -1 # old (first) sentence, last token
            else:
                sequence_lengths0 = -1
                sequence_lengths1 = -1
        
        sequence_end_ix_old = sequence_lengths0
        sequence_end_ix_new = sequence_lengths1
        sequence_end_ix_old = sequence_end_ix_old.to(hidden_states.device)
        sequence_end_ix_new = sequence_end_ix_new.to(hidden_states.device)

        hidden_states_old = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_end_ix_old] 
        hidden_states_new = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_end_ix_new] 

        if self.emb_type == 'diff':
            hidden_states = torch.as_tensor(hidden_states_new - hidden_states_old) 
        elif self.emb_type == 'diffABS':
            hidden_states = torch.abs(torch.as_tensor(hidden_states_new - hidden_states_old))
        elif self.emb_type == 'n-diffABS':
            diff = torch.abs(torch.as_tensor(hidden_states_new - hidden_states_old))
            hidden_states = torch.cat((hidden_states_new, diff),1)
        elif self.emb_type == 'n-diffABS-o':
                diff = torch.abs(torch.as_tensor(hidden_states_new - hidden_states_old))
                hidden_states = torch.cat((hidden_states_new, diff, hidden_states_old),1)
        elif self.emb_type == 'n-o':
            hidden_states = torch.cat((hidden_states_new, hidden_states_old),1)

        hidden_states = hidden_states.to(hidden_states_old.device)
        hidden_states = hidden_states.type(self.score.weight.dtype)
        pooled_logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
