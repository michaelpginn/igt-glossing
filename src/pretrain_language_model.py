import math
import random

import click
import torch
from datasets import DatasetDict
from transformers import RobertaConfig, TrainingArguments, Trainer, RobertaForMaskedLM, DataCollatorForLanguageModeling

import wandb
from data_handling import create_vocab, load_data_file, prepare_dataset_mlm
from tokenizer import WordLevelTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--arch_size", type=str)
@click.option("--project", type=str)
@click.option("--train_data", type=click.Path(exists=True))
@click.option("--eval_data", type=click.Path(exists=True))
@click.option("--position_embeddings", type=str)
def train(arch_size: str = 'micro',
          project: str = 'taxo-morph-pretrain',
          train_data: str = "../data/usp-train-track2-uncovered",
          eval_data: str = "../data/usp-dev-track2-uncovered",
          position_embeddings: str = "absolute"):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64
    EPOCHS = 100

    wandb.init(project=project, entity="michael-ginn", config={
        "bert-size": arch_size,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "position_embeddings": position_embeddings
    })

    random.seed(13)

    train_data = load_data_file(train_data)
    dev_data = load_data_file(eval_data)

    print("Preparing datasets...")

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
    tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=MODEL_INPUT_LENGTH)

    dataset = DatasetDict()
    dataset['train'] = prepare_dataset_mlm(data=[line.morphemes() for line in train_data], tokenizer=tokenizer,
                                           device=device)
    dataset['dev'] = prepare_dataset_mlm(data=[line.morphemes() for line in dev_data], tokenizer=tokenizer,
                                         device=device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")

    if arch_size == 'full':
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=MODEL_INPUT_LENGTH,
            pad_token_id=tokenizer.PAD_ID,
            position_embedding_type=position_embeddings
        )
    else:
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=MODEL_INPUT_LENGTH,
            pad_token_id=tokenizer.PAD_ID,
            num_hidden_layers=3,
            hidden_size=100,
            num_attention_heads=5,
            position_embedding_type=position_embeddings
        )
    language_model = RobertaForMaskedLM(config=config)

    args = TrainingArguments(
        output_dir=f"../pretrain-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=3,
        weight_decay=0,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=False,
        report_to="wandb"
    )
    trainer = Trainer(
        model=language_model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        data_collator=data_collator
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    print(eval_results)

    trainer.save_model(f'../models/usp-mlm-{position_embeddings}-{arch_size}')


if __name__ == "__main__":
    train()
