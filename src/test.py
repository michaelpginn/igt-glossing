from data import load_data_file, create_encoder, prepare_dataset
from taxonomic_loss_model import create_model
from uspanteko_morphology import morphology
from transformers import BertConfig, Trainer, TrainingArguments
import numpy as np
from eval import eval_accuracy

INPUT_LENGTH = 512
data = load_data_file('../data/usp-train-track2-uncovered')
encoder = create_encoder(data, threshold=1)
dataset = prepare_dataset(data=[data[0]], encoder=encoder,
                          model_input_length=INPUT_LENGTH,  device='cpu')


config = BertConfig(
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=INPUT_LENGTH,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[1])
    )
model = create_model(encoder, INPUT_LENGTH)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predicted output
    # print(preds)
    decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=1)
    print('Preds:\t', decoded_preds)

    # Decode (gold) labels
    print(labels)
    decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=1)
    print('Labels:\t', decoded_labels)

    return eval_accuracy(decoded_preds, decoded_labels)


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=2)

args = TrainingArguments(
    output_dir=f"../training-checkpoints",
    evaluation_strategy="steps",
    num_train_epochs=500,
    per_device_train_batch_size=1, # set batch size to 1
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    eval_steps=20,
    logging_dir='./logs'
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()

trainer.predict(test_dataset=dataset)