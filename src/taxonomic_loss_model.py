import torch
from torch import nn
from transformers import RobertaModel, BertConfig, RobertaForTokenClassification, XLMRobertaModel, \
    XLMRobertaForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "mps"


def get_hierarchy_matrix(hierarchy_tree, num_tags, max_depth):
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


def taxonomic_loss(logits, labels, max_depth, hierarchy_matrix, loss_sum):
    labels[labels == -100] = 66
    loss_fct = nn.CrossEntropyLoss()

    all_loss = None

    for level in range(max_depth):
        # Create groups for this level of hierarchy with indices
        groups = list(hierarchy_matrix.groupby(by=level).groups.values())

        # For each group, sum the logits for all items
        group_logits = torch.transpose(torch.stack([logits[:, group].sum(axis=1) for group in groups]), 0, 1)

        # Use the hierarchy matrix to turn node labels into group labels
        group_labels = np.vstack([hierarchy_matrix, [-100] * hierarchy_matrix.shape[1]])[
            labels.cpu(), level]
        group_labels = torch.tensor(group_labels, dtype=torch.long).to(device)

        # Calculate crossentropy loss between group logits and group labels
        level_loss = loss_fct(group_logits.to(device), group_labels.to(device))

        if loss_sum == 'harmonic':
            divisor = max_depth - level
            level_loss = torch.div(level_loss, divisor)

        # print("LEVEL", level, level_loss)
        if all_loss is not None:
            all_loss += level_loss
        else:
            all_loss = level_loss
    return all_loss


class TaxonomicLossModel(RobertaForTokenClassification):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, loss_sum='linear'):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.morphology = None
        self.hierarchy_matrix = None
        self.max_depth = None
        self.loss_sum = loss_sum

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def use_morphology_tree(self, tree, max_depth: int):
        self.morphology = tree
        self.hierarchy_matrix = pd.DataFrame(get_hierarchy_matrix(self.morphology, self.num_labels, max_depth))
        self.max_depth = max_depth

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

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = taxonomic_loss(logits.view(-1, self.num_labels), labels.view(-1), self.max_depth,
                                  self.hierarchy_matrix, self.loss_sum)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMTaxonomicLossModel(XLMRobertaForTokenClassification):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, loss_sum='linear'):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.morphology = None
        self.hierarchy_matrix = None
        self.max_depth = None
        self.loss_sum = loss_sum

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def use_morphology_tree(self, tree, max_depth: int):
        self.morphology = tree
        self.hierarchy_matrix = pd.DataFrame(get_hierarchy_matrix(self.morphology, self.num_labels, max_depth))
        self.max_depth = max_depth

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

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = taxonomic_loss(logits.view(-1, self.num_labels), labels.view(-1), self.max_depth,
                                  self.hierarchy_matrix, self.loss_sum)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
