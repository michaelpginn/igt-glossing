import click
import wandb
import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, RobertaForTokenClassification
from datasets import DatasetDict
from typing import Optional
from data import prepare_dataset, load_data_file, create_vocab, create_gloss_vocab
from encoder import CustomEncoder
from eval import eval_accuracy
from taxonomic_loss_model import TaxonomicLossModel
from uspanteko_morphology import morphology as full_morphology_tree, simplified_morphology as simplified_morphology_tree
import random


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_trainer(model: RobertaForTokenClassification, dataset: Optional[DatasetDict], encoder: CustomEncoder, batch_size, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        decoded_preds = encoder.batch_decode(preds, vocab='output')

        # Decode (gold) labels
        decoded_labels = encoder.batch_decode(labels, vocab='output')

        decoded_preds = [preds[:len(labels)] for preds, labels in zip(decoded_preds, decoded_labels)]
        print('Preds:\t', decoded_preds[0])
        print('Labels:\t', decoded_labels[0])

        return eval_accuracy(decoded_preds, decoded_labels)

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument('loss', type=click.Choice(['flat', 'tax', 'tax_simple'], case_sensitive=False))
@click.option("--train_size", help="Number of items to sample from the training data", type=int)
@click.option("--seed", help="Random seed", type=int)
def train(loss: str, train_size: int, seed: int):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64
    EPOCHS = 100

    run_name = f"{train_size if train_size else 'full'}-{loss}-{seed}"

    wandb.init(project="taxo-morph-finetuning-ignore", entity="michael-ginn", name=run_name, config={
        "loss": loss,
        "train-size": train_size if train_size else "full",
        "random-seed": seed,
    })

    random.seed(seed)

    morphology_tree = simplified_morphology_tree if loss == 'tax_simple' else full_morphology_tree

    train_data = load_data_file(f"../data/usp-train-track2-uncovered")
    dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)

    if train_size:
        train_data = random.sample(train_data, train_size)

    glosses = create_gloss_vocab(morphology_tree)
    encoder = CustomEncoder(vocabulary=train_vocab, output_vocabulary=glosses)

    dataset = DatasetDict()
    dataset['train'] = prepare_dataset(data=train_data, encoder=encoder, model_input_length=MODEL_INPUT_LENGTH, device=device)
    dataset['dev'] = prepare_dataset(data=dev_data, encoder=encoder, model_input_length=MODEL_INPUT_LENGTH, device=device)

    if loss == "flat":
        model = AutoModelForTokenClassification.from_pretrained("michaelginn/uspanteko-roberta-base", num_labels=len(glosses))
    elif loss == "tax" or loss == "tax_simple":
        model = TaxonomicLossModel.from_pretrained("michaelginn/uspanteko-roberta-base", num_labels=len(glosses))
        model.use_morphology_tree(morphology_tree, max_depth=1 if loss == 'tax_simple' else 5)

    trainer = create_trainer(model, dataset=dataset, encoder=encoder, batch_size=BATCH_SIZE, max_epochs=EPOCHS)
    trainer.train()
    trainer.save_model(f'./{run_name}')


if __name__ == "__main__":
    cli()