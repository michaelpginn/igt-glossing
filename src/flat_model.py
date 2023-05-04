import torch
from transformers import BertForTokenClassification, BertConfig, BertPreTrainedModel
from encoder import MultiVocabularyEncoder, special_chars


def create_model(encoder: MultiVocabularyEncoder, sequence_length) -> BertPreTrainedModel:
    """Creates the appropriate model"""
    print("Creating model...")
    config = BertConfig(
        vocab_size=len(encoder.vocabularies[0]) + len(encoder.special_chars),
        # num_hidden_layers=8,
        # num_attention_heads=8,
        # hidden_size=512,
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[1])
    )
    model = BertForTokenClassification(config)
    print(model.config)
    return model