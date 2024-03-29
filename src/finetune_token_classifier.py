import random
from typing import Optional, List

import click
import torch
import wandb
from datasets import DatasetDict
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, RobertaForTokenClassification, \
    RobertaForMaskedLM, RobertaConfig

from data_handling import prepare_dataset, load_data_file, create_vocab, create_gloss_vocab, prepare_multitask_dataset
from eval import eval_accuracy
from multistage_model import MultistageModel
from multitask_model import MultitaskModel
from taxonomic_loss_model import TaxonomicLossModel
from tokenizer import WordLevelTokenizer
from uspanteko_morphology import morphology as full_morphology_tree
from denoised_model import DenoisedModel

device = "cuda:0" if torch.cuda.is_available() else "mps"


def create_trainer(model: RobertaForTokenClassification, dataset: Optional[DatasetDict], tokenizer: WordLevelTokenizer,
                   labels: List[str], batch_size, max_epochs, weight_decay, report_to):
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
        decoded_labels = [[labels[index] for index in label_seq if len(labels) > index >= 0] for label_seq in
                          gold_labels]

        # Trim preds to the same length as the labels
        decoded_preds = [pred_seq[:len(label_seq)] for pred_seq, label_seq in zip(decoded_preds, decoded_labels)]

        print('Preds:\t', decoded_preds[0])
        print('Labels:\t', decoded_labels[0])

        return eval_accuracy(decoded_preds, decoded_labels)

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../finetune-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        weight_decay=weight_decay,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        load_best_model_at_end=False,
        logging_strategy='epoch',
        report_to=report_to
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
@click.option('--model_type',
              type=click.Choice(['flat', 'multitask', 'multistage', 'tax_loss', 'harmonic_loss', 'denoised',
                                 'relative_position_embeddings', 'no_pretrained']))
@click.option("--train_size", help="Number of items to sample from the training data", type=int)
@click.option("--train_data", type=click.Path(exists=True))
@click.option("--eval_data", type=click.Path(exists=True))
@click.option("--seed", help="Random seed", type=int)
@click.option("--additional_train_data", type=click.Path(exists=True))
@click.option("--epochs", help="Max # epochs", type=int)
@click.option("--weight_decay", help="Fraction of weight decay", type=float)
@click.option("--pretrained", type=str)
@click.option("--project", type=str)
def train(model_type: str, train_size: int, seed: int,
          train_data: str = "../data/usp-train-track2-uncovered",
          eval_data: str = "../data/usp-dev-track2-uncovered",
          additional_train_data: str = None,
          epochs: int = 200,
          weight_decay: float = 0.1,
          pretrained: str = None,
          project: str = 'taxo-morph-finetune'):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64

    run_name = f"{train_size if train_size else 'full'}-{model_type}-{seed}{'-finetune' if pretrained is not None else ''}"
    
    base_model = pretrained if pretrained is not None else "../models/usp-mlm-absolute-micro"

    wandb.init(project=project, entity="michael-ginn", name=run_name, config={
        "train-size": train_size if train_size else "full",
        "random-seed": seed,
        "type": model_type,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "base_model": base_model
    })

    random.seed(seed)

    morphology_tree = full_morphology_tree

    train_data = load_data_file(train_data)
    dev_data = load_data_file(eval_data)

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
    tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=MODEL_INPUT_LENGTH)

    if train_size:
        train_data = random.sample(train_data, train_size)

    glosses = create_gloss_vocab(morphology_tree)

    dataset = DatasetDict()
    
    if additional_train_data is not None:
        additional_train_data = load_data_file(additional_train_data)
        train_data += additional_train_data
    
    if model_type == 'multitask' or model_type == 'multistage':
        dataset['train'] = prepare_multitask_dataset(data=train_data, tokenizer=tokenizer, labels=glosses,
                                                     device=device)
        dataset['dev'] = prepare_multitask_dataset(data=dev_data, tokenizer=tokenizer, labels=glosses, device=device)
    else:
        dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizer, labels=glosses, device=device)
        dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizer, labels=glosses, device=device)

    if model_type != 'multistage':
        if model_type == 'multitask':
            model = MultitaskModel.from_pretrained(base_model,
                                                   classifier_head_sizes=[66, 21, 19, 10])
        elif model_type == 'flat':
            model = AutoModelForTokenClassification.from_pretrained(base_model,
                                                                    num_labels=len(glosses))
        elif model_type == 'denoised':
            model = DenoisedModel.from_pretrained(base_model,
                                                  num_labels=len(glosses))
        elif model_type == 'relative_position_embeddings':
            model = AutoModelForTokenClassification.from_pretrained("../models/usp-mlm-relative_key_query-full",
                                                                    num_labels=len(glosses))
        elif model_type == 'no_pretrained':
            config = RobertaConfig(
                vocab_size=tokenizer.vocab_size,
                max_position_embeddings=MODEL_INPUT_LENGTH,
                pad_token_id=tokenizer.PAD_ID,
                num_hidden_layers=3,
                hidden_size=100,
                num_attention_heads=5,
                position_embedding_type='absolute'
            )
            model = RobertaForMaskedLM(config=config)
        else:
            if model_type == 'tax_loss':
                model = TaxonomicLossModel.from_pretrained(base_model, num_labels=len(glosses),
                                                           loss_sum='linear')
            elif model_type == 'harmonic_loss':
                model = TaxonomicLossModel.from_pretrained(base_model, num_labels=len(glosses),
                                                           loss_sum='harmonic')
            model.use_morphology_tree(morphology_tree, max_depth=5)

        trainer = create_trainer(model, dataset=dataset, tokenizer=tokenizer, labels=glosses, batch_size=BATCH_SIZE,
                                 max_epochs=epochs, weight_decay=weight_decay, report_to='wandb')
        trainer.train()
    else:
        # Train in multiple stages
        model = MultistageModel.from_pretrained(base_model,
                                                classifier_head_sizes=[66, 21, 19, 10])

        for stage in [3, 2, 1, 0]:
            model.current_stage = stage
            trainer = create_trainer(model, dataset=dataset, tokenizer=tokenizer, labels=glosses, batch_size=BATCH_SIZE,
                                     max_epochs=epochs, weight_decay=weight_decay,
                                     report_to='wandb' if stage == 0 else "none")
            trainer.train()

    trainer.save_model(f'../models/{run_name}-{weight_decay}wd')


if __name__ == "__main__":
    cli()
