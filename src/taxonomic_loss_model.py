import torch
from torch import nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from encoder import MultiVocabularyEncoder, special_chars
from uspanteko_morphology import morphology

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaxonomicLossModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, morphology):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.morphology = morphology

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.output_layer = nn.Linear(config.hidden_size, self.num_labels)

        self.hierarchy_matrix = pd.DataFrame(self.get_hierarchy_matrix(morphology, 66, 5))

        # Initialize weights and apply final processing
        self.post_init()

    def get_hierarchy_matrix(self, hierarchy_tree, num_tags, max_depth):
        """Takes a hierarchical tree, and creates a matrix of a_i,j where i is the tag and j is the level of hierarchy.
        """
        matrix = np.zeros((num_tags, max_depth))

        def parse_tree(tree, group_prefix=[], start_tag=0):
            for group_index, item in enumerate(tree):
                if start_tag == 0:
                    start_tag += 1
                    continue
                if isinstance(item, tuple):
                    start_tag = parse_tree(item[1],
                                           group_prefix=group_prefix + [matrix[start_tag - 1][len(group_prefix)] + 1],
                                           start_tag=start_tag)
                else:
                    matrix[start_tag] = matrix[start_tag - 1] + 1
                    matrix[start_tag][:len(group_prefix)] = group_prefix
                    start_tag += 1
            return start_tag

        parse_tree(hierarchy_tree)
        return matrix


    def taxonomic_loss(self, logits, labels):
        labels[labels == -100] = 66
        loss_fct = nn.CrossEntropyLoss()

        all_loss = None

        for level in range(5):
            groups = list(self.hierarchy_matrix.groupby(by=level).groups.values())
            group_logits = torch.transpose(torch.stack([logits[:,group].sum(axis=1) for group in groups]), 0, 1)
            group_labels = torch.LongTensor(np.vstack([self.hierarchy_matrix, [-100] * self.hierarchy_matrix.shape[1]])[labels, level])
            level_loss = loss_fct(group_logits, group_labels)
            print('LEVEL', level, level_loss)
            if all_loss is not None:
                all_loss += level_loss
            else:
                all_loss = level_loss
        return all_loss

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

        outputs = self.bert(
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
        logits = self.output_layer(sequence_output)

        loss = None
        if labels is not None:
            print('NORMAL', nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1)))

            loss = self.taxonomic_loss(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_model(encoder: MultiVocabularyEncoder, sequence_length) -> BertPreTrainedModel:
    """Creates the appropriate model"""
    print("Creating model...")
    config = BertConfig(
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[1])
    )
    model = TaxonomicLossModel(config, morphology)
    print(model.config)
    return model