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

    
class LlamaForSequenceClassificationSiamese(LlamaPreTrainedModel):
    def __init__(self, config, model=None, score=None,  emb_type=None):
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs_old = self.model(
            torch.squeeze(input_ids[:,0,:,:],1),
            attention_mask=torch.squeeze(attention_mask[:,0,:,:],1),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        transformer_outputs_new = self.model(
            torch.squeeze(input_ids[:,1,:,:],1),
            attention_mask=torch.squeeze(attention_mask[:,1,:,:],1),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        if input_ids is not None:
            batch_size = torch.squeeze(input_ids[:,0,:,:],1).shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths_old = -1
            sequence_lengths_new = -1
        else:
            if input_ids is not None:
                sequence_lengths_old = (torch.eq(torch.squeeze(input_ids[:,0,:,:],1), self.config.pad_token_id).long().argmax(-1) - 1)
                sequence_lengths_new = (torch.eq(torch.squeeze(input_ids[:,1,:,:],1), self.config.pad_token_id).long().argmax(-1) - 1)
            else:
                sequence_lengths_old = -1
                sequence_lengths_new = -1

        hidden_states_old = transformer_outputs_old[0]
        hidden_states_new = transformer_outputs_new[0]
        sequence_lengths_old = sequence_lengths_old.to(hidden_states_old.device)
        sequence_lengths_new = sequence_lengths_new.to(hidden_states_new.device)
        # get the last token embedding as sentence embedding
        hidden_states_old = hidden_states_old[torch.arange(batch_size), sequence_lengths_old]
        hidden_states_new = hidden_states_new[torch.arange(batch_size), sequence_lengths_new]
            

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
        logits = self.score(hidden_states)
        pooled_logits = logits
        

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
            output = (pooled_logits,) + transformer_outputs_new[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs_new.past_key_values,
            hidden_states=transformer_outputs_new.hidden_states,
            attentions=transformer_outputs_new.attentions,
        )


