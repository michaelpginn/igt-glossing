import math
import random

import click
import torch
from datasets import DatasetDict
from transformers import RobertaConfig, TrainingArguments, Trainer, RobertaForMaskedLM, DataCollatorForLanguageModeling

import wandb
from data_handling import load_data_file, prepare_dataset_mlm, create_vocab, create_gloss_vocab
from tokenizer import WordLevelTokenizer
from uspanteko_morphology import morphology

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--arch_size", type=str)
@click.option("--project", type=str)
@click.option("--train_data", type=click.Path(exists=True))
@click.option("--eval_data", type=click.Path(exists=True))
def train(arch_size: str = 'micro',
          project: str = 'taxo-morph-train-denoiser',
          train_data: str = "../data/usp-train-track2-uncovered",
          eval_data: str = "../data/usp-dev-track2-uncovered"):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64
    EPOCHS = 100

    wandb.init(project=project, entity="michael-ginn", config={
        "bert-size": arch_size,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    })

    random.seed(1)

    train_data = load_data_file(train_data)
    dev_data = load_data_file(eval_data)

    print("Preparing datasets...")

    glosses = create_gloss_vocab(morphology)
    tokenizer = WordLevelTokenizer(vocab=glosses, model_max_length=MODEL_INPUT_LENGTH)

    dataset = DatasetDict()
    dataset['train'] = prepare_dataset_mlm(data=[line.gloss_list(segmented=True) for line in train_data],
                                           tokenizer=tokenizer,
                                           device=device)
    dataset['dev'] = prepare_dataset_mlm(data=[line.gloss_list(segmented=True) for line in dev_data],
                                         tokenizer=tokenizer,
                                         device=device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")

    if arch_size == 'full':
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=MODEL_INPUT_LENGTH,
            pad_token_id=tokenizer.PAD_ID,
        )
    else:
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=MODEL_INPUT_LENGTH,
            pad_token_id=tokenizer.PAD_ID,
            num_hidden_layers=3,
            hidden_size=100,
            num_attention_heads=5
        )
    language_model = RobertaForMaskedLM(config=config)

    args = TrainingArguments(
        output_dir=f"../denoiser-training-checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=EPOCHS,
        report_to="wandb",
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

    trainer.save_model(f'../models/usp-gloss-denoiser')


if __name__ == "__main__":
    train()
