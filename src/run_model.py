import click
import wandb
import torch
import numpy as np
from transformers import BertPreTrainedModel, TrainingArguments, Trainer, BertForTokenClassification
from datasets import DatasetDict
from typing import Optional
from data import prepare_dataset, load_data_file, create_encoder, write_predictions
from encoder import MultiVocabularyEncoder, special_chars, load_encoder
from flat_model import create_model as create_flat_model
from taxonomic_loss_model import create_model as create_taxonomic_model
from eval import eval_morpheme_glosses

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_trainer(model: BertPreTrainedModel, dataset: Optional[DatasetDict], encoder: MultiVocabularyEncoder, batch_size, lr, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        print(preds)
        decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=1)
        print(decoded_preds[0:1])

        # Decode (gold) labels
        print(labels)
        # labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=1)
        print(decoded_labels[0:1])

        return eval_morpheme_glosses(pred_morphemes=decoded_preds, gold_morphemes=decoded_labels)


    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["dev"] if dataset else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    return trainer


@click.command()
@click.argument('mode')
@click.argument('model')
@click.option("--pretrained_path", help="Path to pretrained model", type=click.Path(exists=True))
@click.option("--encoder_path", help="Path to pretrained encoder", type=click.Path(exists=True))
@click.option("--data_path", help="The dataset to run predictions on. Only valid in predict mode.", type=click.Path(exists=True))
def main(mode: str, model: str, pretrained_path: str, encoder_path: str, data_path: str):
    if mode == 'train':
        wandb.init(project="struct-morph", entity="michael-ginn")

    MODEL_INPUT_LENGTH = 512

    train_data = load_data_file(f"../data/usp-train-track2-uncovered")
    dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

    print("Preparing datasets...")

    if mode == 'train':
        encoder = create_encoder(train_data, threshold=1)
        encoder.save()
        dataset = DatasetDict()
        dataset['train'] = prepare_dataset(data=train_data, encoder=encoder, model_input_length=MODEL_INPUT_LENGTH, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, encoder=encoder, model_input_length=MODEL_INPUT_LENGTH, device=device)

        create_model = create_flat_model if model == 'flat' else create_taxonomic_model
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH).to(device)
        trainer = create_trainer(model, dataset=dataset, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=30)

        print("Training...")
        trainer.train()
        print("Saving model to ./output")
        trainer.save_model('./output')
        print("Model saved at ./output")
    # elif mode == 'predict':
        # encoder = load_encoder(encoder_path)
        # if not hasattr(encoder, 'segmented'):
        #     encoder.segmented = True
        # print("ENCODER SEGMENTING INPUT: ", encoder.segmented)
        # predict_data = load_data_file(data_path)
        # predict_data = prepare_dataset(data=predict_data, encoder=encoder,
        #                                model_input_length=MODEL_INPUT_LENGTH, device=device)
        # model = BertForTokenClassification.from_pretrained(pretrained_path)
        # trainer = create_trainer(model, dataset=None, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=50)
        # preds = trainer.predict(test_dataset=predict_data).predictions
        # write_predictions(data_path, preds=preds, pred_input_data=predict_data, encoder=encoder, from_vocabulary_index=2)


if __name__ == "__main__":
    main()


# TODO:
# - Ignore divider tokens in loss calculation and eval
# - Fix encoder vocab for glosses for hierarchical model