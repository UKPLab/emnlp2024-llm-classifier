# Built upon the huggingface implementation 
from __future__ import annotations
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
import importlib
import os
from typing import Optional
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoTokenizer

from peft.config import PeftConfig
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.peft_model import PeftModel
from peft.utils.constants import TOKENIZER_CONFIG_NAME
from peft.utils.other import check_file_exists_on_hf_hub

from peft.tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    OFTModel,
    PolyModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from peft.utils import (
    PeftType,
    _get_batch_size,
    _set_trainable,
)

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
    PeftType.IA3: IA3Model,
    PeftType.OFT: OFTModel,
    PeftType.POLY: PolyModel,
}


class PeftModelForSequenceClassificationSiamese(PeftModel):
    """
    Peft model for sequence classification tasks with the snet approach.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    """

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
       
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})


        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                if peft_config.peft_type == PeftType.POLY:
                    kwargs["task_ids"] = task_ids
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )



class _BaseAutoPeftModelSiamese:
    _target_class = None
    _target_peft_class = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(  
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        emb_type=None,
        **kwargs,
    ):
        r"""
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        if cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            # this is only in the case where we use `AutoPeftModel`
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )

        if task_type is not None:
            expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
            if cls._target_peft_class.__name__ != expected_target_class.__name__:
                raise ValueError(
                    f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__ }"
                    " make sure that you are loading the correct model for your task type."
                )
        elif task_type is None and getattr(peft_config, "auto_mapping", None) is not None:
            auto_mapping = getattr(peft_config, "auto_mapping", None)
            base_model_class = auto_mapping["base_model_class"]
            parent_library_name = auto_mapping["parent_library"]

            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
            )
        

        base_model = target_class.from_pretrained(base_model_path, emb_type=emb_type, **kwargs) 
        

        tokenizer_exists = False
        if os.path.exists(os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)):
            tokenizer_exists = True
        else:
            token = kwargs.get("token", None)
            if token is None:
                token = kwargs.get("use_auth_token", None)

            tokenizer_exists = check_file_exists_on_hf_hub(
                repo_id=pretrained_model_name_or_path,
                filename=TOKENIZER_CONFIG_NAME,
                revision=kwargs.get("revision", None),
                repo_type=kwargs.get("repo_type", None),
                token=token,
            )

        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False)
            )
            base_model.resize_token_embeddings(len(tokenizer))

        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )

from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
class AutoModelForSequenceClassificationSiamese(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class AutoPeftModelForSequenceClassificationSiamese(_BaseAutoPeftModelSiamese):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_type = args.emb_type
    _target_class = AutoModelForSequenceClassificationSiamese
    _target_peft_class = PeftModelForSequenceClassificationSiamese
    

