import click
import wandb
import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, RobertaForTokenClassification
from datasets import DatasetDict
from typing import Optional, List
from data import prepare_dataset, load_data_file, create_vocab, create_gloss_vocab, prepare_multitask_dataset
from eval import eval_accuracy
from taxonomic_loss_model import TaxonomicLossModel
from uspanteko_morphology import morphology as full_morphology_tree, simplified_morphology as simplified_morphology_tree
import random
from tokenizer import WordLevelTokenizer
from multitask_model import MultitaskModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_trainer(model: RobertaForTokenClassification, dataset: Optional[DatasetDict], tokenizer: WordLevelTokenizer, labels: List[str], batch_size, max_epochs):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, gold_labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        print("PREDS", preds)
        print("LABELS", gold_labels)
        if len(gold_labels.shape) > 2:
            gold_labels = gold_labels.take(axis=1, indices=0)

        print(gold_labels.shape)

        # Decode predicted output
        decoded_preds = [[labels[index] for index in pred_seq if len(labels) > index >= 0] for pred_seq in preds]

        # Decode (gold) labels
        decoded_labels = [[labels[index] for index in label_seq if len(labels) > index >= 0] for label_seq in gold_labels]

        # Trim preds to the same length as the labels
        decoded_preds = [pred_seq[:len(label_seq)] for pred_seq, label_seq in zip(decoded_preds, decoded_labels)]

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
@click.option('--multitask', type=bool)
@click.option('--loss_sum', type=click.Choice(['linear', 'harmonic'], case_sensitive=False))
@click.option("--train_size", help="Number of items to sample from the training data", type=int)
@click.option("--seed", help="Random seed", type=int)
def train(loss: str, train_size: int, loss_sum: str, multitask: bool, seed: int):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64
    EPOCHS = 30

    run_name = f"{train_size if train_size else 'full'}-{'multi' if multitask else 'single'}-{seed}"

    wandb.init(project="taxo-morph-finetuning-v3", entity="michael-ginn", name=run_name, config={
        # "loss": loss + '-' + loss_sum,
        "train-size": train_size if train_size else "full",
        "random-seed": seed,
        "multitask": multitask
    })

    random.seed(seed)

    morphology_tree = simplified_morphology_tree if loss == 'tax_simple' else full_morphology_tree

    train_data = load_data_file(f"../data/usp-train-track2-uncovered")
    dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
    tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=MODEL_INPUT_LENGTH)

    if train_size:
        train_data = random.sample(train_data, train_size)

    glosses = create_gloss_vocab(morphology_tree)

    dataset = DatasetDict()

    if not multitask:
        dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizer, labels=glosses, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizer, labels=glosses, device=device)
    else:
        dataset['train'] = prepare_multitask_dataset(data=train_data, tokenizer=tokenizer, labels=glosses, device=device)
        dataset['dev'] = prepare_multitask_dataset(data=dev_data, tokenizer=tokenizer, labels=glosses, device=device)

    # if loss == "flat":
    #     model = AutoModelForTokenClassification.from_pretrained("michaelginn/uspanteko-mlm-large", num_labels=len(glosses))
    # elif loss == "tax" or loss == "tax_simple":
    #     model = TaxonomicLossModel.from_pretrained("michaelginn/uspanteko-mlm-large", num_labels=len(glosses), loss_sum=loss_sum)
    #     model.use_morphology_tree(morphology_tree, max_depth=2 if loss == 'tax_simple' else 5)
    # else:
    #     raise ValueError("Invalid loss provided.")

    if multitask:
        model = MultitaskModel.from_pretrained("michaelginn/uspanteko-mlm-large", classifier_head_sizes=[66, 21, 19, 10])
    else:
        model = AutoModelForTokenClassification.from_pretrained("michaelginn/uspanteko-mlm-large", num_labels=len(glosses))

    trainer = create_trainer(model, dataset=dataset, tokenizer=tokenizer, labels=glosses, batch_size=BATCH_SIZE, max_epochs=EPOCHS)
    trainer.train()
    trainer.save_model(f'../models/{run_name}')


if __name__ == "__main__":
    cli()