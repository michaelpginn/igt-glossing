import torch
from torch import nn
from transformers import RobertaModel, BertConfig, RobertaForTokenClassification, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MultistageModel(RobertaForTokenClassification):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, classifier_head_sizes):
        """`output_heads` should be a list of integers, where each number is the number of labels for a given head"""
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier_head_sizes = classifier_head_sizes
        self.classifier_heads = [nn.Linear(config.hidden_size, head_size).to(device) for head_size in classifier_head_sizes]

        # What head should we be training? Update this for each stage of training
        self.current_stage = len(classifier_head_sizes) - 1

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_classifier_heads)`, *optional*):
            Labels for computing the token classification loss.`.
        """
        print(f"Training at stage {self.current_stage}")
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output.to(device)

        # Get logits for all classifier heads
        all_classifier_logits = []
        for head in self.classifier_heads:
            logits = head(sequence_output)
            all_classifier_logits.append(logits)

        # Compute loss and sum across heads
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            labels = labels.to(all_classifier_logits[0].device)

            level_labels = labels.select(1, self.current_stage).reshape(-1)
            logits = all_classifier_logits[self.current_stage]
            loss = loss_fct(logits.view(-1, self.classifier_head_sizes[self.current_stage]), level_labels)

        # We just return the first logits, which are our target for evaluation
        if not return_dict:
            output = (all_classifier_logits[0],) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=all_classifier_logits[0],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )