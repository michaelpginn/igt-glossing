import torch
from torch import nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Union, Tuple
from encoder import MultiVocabularyEncoder, special_chars
from uspanteko_morphology import morphology

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class HierarchicalMorphemeLabelingModel(BertPreTrainedModel):
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
        self.pos_layer = nn.Linear(config.hidden_size, len(morphology))

        # Will contain tuples: (layer, children)
        # layer: the actual nn layer
        # children: a recursive list of tuples if any
        self.layer_hierarchy = []

        def create_subtree(morphology_subtree, layer_hierarchy):
            for index, item in enumerate(morphology_subtree):
                if isinstance(item, tuple):
                    subtree_layer = nn.Linear(1, len(item[1])).to(device)
                    subtree_item = (subtree_layer, [])
                    layer_hierarchy.append(subtree_item)
                    create_subtree(item[1], subtree_item[1])
                else:
                    layer_hierarchy.append(None)

        create_subtree(morphology, self.layer_hierarchy)

        # Initialize weights and apply final processing
        self.post_init()


    def evaluate_classification_head(self, sequence_output):
        """Takes the output from the BERT embedder and runs it through the hierarchical tree network"""
        print(sequence_output.size())
        pos_output = self.pos_layer(sequence_output)
        print(pos_output.size())
        def evaluate_recursive(inputs, layer_hierarchy):
            """Inputs is (batch size x #)"""
            print(inputs.size())
            outputs = []
            for i in range(inputs.size(dim=2)):
                if layer_hierarchy[i] is not None:
                    next_output_layer = layer_hierarchy[i][0]
                    next_output = evaluate_recursive(next_output_layer(inputs[:,:,i:i+1]), layer_hierarchy[i][1])
                    outputs.append(next_output)
                else:
                    outputs.append(inputs[:,:,i:i+1])
            return torch.cat(outputs, dim=2).to(device)

        return evaluate_recursive(pos_output, self.layer_hierarchy)


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
        logits = self.evaluate_classification_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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
        num_labels=len(encoder.vocabularies[1]) + len(special_chars)
    )
    model = HierarchicalMorphemeLabelingModel(config, morphology)
    print(model.config)
    return model