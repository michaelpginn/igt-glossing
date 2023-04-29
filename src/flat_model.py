import torch
from transformers import BertForTokenClassification, BertConfig, BertPreTrainedModel
from encoder import MultiVocabularyEncoder, special_chars


def create_model(encoder: MultiVocabularyEncoder, sequence_length) -> BertPreTrainedModel:
    """Creates the appropriate model"""
    print("Creating model...")
    config = BertConfig(
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[1]) + len(special_chars)
    )
    model = BertForTokenClassification(config)
    print(model.config)
    return model