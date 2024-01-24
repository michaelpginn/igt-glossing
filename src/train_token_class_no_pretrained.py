import random
from typing import Optional, List

import click
import torch
import wandb
from datasets import DatasetDict, load_dataset
import pandas as pd
from transformers import (
    TrainerCallback,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    RobertaForTokenClassification,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    DataCollatorForTokenClassification
)

from data_handling import prepare_dataset, create_vocab, create_gloss_vocab, split_line
from eval import compute_metrics
from taxonomic_loss_model import TaxonomicLossModel, XLMTaxonomicLossModel, get_hierarchy_matrix
from tokenizer import WordLevelTokenizer
from uspanteko_morphology import morphology as full_morphology_tree

device = "cuda:0" if torch.cuda.is_available() else "mps"


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only start applying early stopping logic after start_epoch
        if state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model_type',
              type=click.Choice(['flat', 'tax_loss', 'harmonic_loss']))
@click.option("--arch", type=click.Choice(['roberta', 'xlm-roberta']))
@click.option("--seed", help="Random seed", type=int)
@click.option("--epochs", help="Max # epochs", type=int)
@click.option("--project", type=str, default='taxo-morph-naacl')
def train(model_type: str, seed: int,
          arch: str = "roberta",
          epochs: int = 200,
          project: str = 'taxo-morph-naacl'):
    BATCH_SIZE = 64

    run_name = f"{model_type}-{arch}-{seed}"

    wandb.init(project=project, entity="michael-ginn", name=run_name, config={
        "random-seed": seed,
        "type": model_type,
        "epochs": epochs,
        "arch": arch,
    })

    random.seed(seed)

    morphology_tree = full_morphology_tree
    glosses = create_gloss_vocab(morphology_tree)
    hierarchy_matrix = pd.DataFrame(get_hierarchy_matrix(morphology_tree, len(glosses), 5))

    dataset = load_dataset('lecslab/usp-igt', download_mode="force_redownload")

    def segment(row):
        row["morphemes"] = split_line(row["segmentation"], prefix="usp_")
        row["glosses"] = split_line(row["pos_glosses"], prefix=None)
        return row

    dataset = dataset.map(segment, load_from_cache_file=False)
    train_vocab = create_vocab(dataset['train']['morphemes'], threshold=1)

    if arch == 'roberta':
        # No pretrained model, start from random
        tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=64)

        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=512,
            pad_token_id=tokenizer.PAD_ID,
            position_embedding_type='absolute',
            num_labels=len(glosses)
        )
    elif arch == 'xlm-roberta':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base', model_max_length=64)
        tokenizer.add_tokens(train_vocab)
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base', num_labels=len(glosses))

    dataset = prepare_dataset(dataset, train_vocab, tokenizer, glosses)

    if arch == 'roberta':
        if model_type == 'flat':
            model = RobertaForTokenClassification(config=config)
        elif model_type == 'tax_loss':
            model = TaxonomicLossModel(config=config, loss_sum='linear')
            model.use_morphology_tree(morphology_tree, max_depth=5)
        elif model_type == 'harmonic_loss':
            model = TaxonomicLossModel(config=config, loss_sum='harmonic')
            model.use_morphology_tree(morphology_tree, max_depth=5)
    elif arch == 'xlm-roberta':
        if model_type == 'flat':
            model = XLMRobertaForTokenClassification(config=config)
        elif model_type == 'tax_loss':
            model = XLMTaxonomicLossModel(config=config, loss_sum='linear')
            model.use_morphology_tree(morphology_tree, max_depth=5)
        elif model_type == 'harmonic_loss':
            model = XLMTaxonomicLossModel(config=config, loss_sum='harmonic')
            model.use_morphology_tree(morphology_tree, max_depth=5)
        # Since we use a pretrained model, we need to update with our added vocabulary
        model.resize_token_embeddings(len(tokenizer))

    def preprocess_logits_for_metrics(logits, _):
        return logits.argmax(dim=2)

    args = TrainingArguments(
        output_dir=f"../finetune-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=3,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        logging_strategy='epoch',
        report_to='wandb'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["eval"] if dataset else None,
        compute_metrics=compute_metrics(glosses, hierarchy_matrix),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            LogCallback,
            DelayedEarlyStoppingCallback(early_stopping_patience=3)
        ],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
    )

    trainer.train()

    trainer.save_model(f'../models/{run_name}')

    test_eval = trainer.evaluate(dataset['test'])
    test_eval = {k.replace('eval', 'test'): test_eval[k] for k in test_eval}
    wandb.log(test_eval)


if __name__ == "__main__":
    cli()
