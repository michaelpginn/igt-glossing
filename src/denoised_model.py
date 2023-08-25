import torch
from torch import nn
from transformers import RobertaModel, BertConfig, RobertaForTokenClassification, AutoModelForMaskedLM
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DenoisedModel(RobertaForTokenClassification):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.denoiser = AutoModelForMaskedLM.from_pretrained("michaelginn/usp-gloss-denoiser")

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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
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
        logits = self.classifier(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Get predictions from logits
        preds = logits.max(-1).indices

        print("Preds before denoising", preds)

        # Increase all input ids (except SEP token) by 4 to account for special tokens
        preds[preds != 1] = preds[preds != 1] + 4

        # Replace any glosses for unknown tokens with MASK
        preds[input_ids == 0] = 3

        # Cut off end of sequence (always garbage)
        preds = preds.narrow(-1, 0, 60)
        attention_mask = attention_mask.narrow(-1, 0, 60)

        # Run denoiser model on preds
        denoised_logits = self.denoiser.forward(input_ids=preds, attention_mask=attention_mask).logits
        denoised_logits = denoised_logits.narrow(-1, 4, 64)

        print("Preds after denoising", denoised_logits.max(-1).indices)

        if not return_dict:
            output = (denoised_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=denoised_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
