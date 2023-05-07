import click
import wandb
import torch
import math
from transformers import RobertaConfig, TrainingArguments, Trainer, RobertaForMaskedLM, DataCollatorForLanguageModeling
from datasets import DatasetDict
from data import prepare_dataset, load_data_file, prepare_dataset_mlm
from encoder import CustomEncoder, create_vocab
from custom_tokenizers import WordLevelTokenizer
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@click.command()
@click.option("--arch_size", type=str)
def train(arch_size: str='micro'):
    MODEL_INPUT_LENGTH = 64
    BATCH_SIZE = 64
    EPOCHS = 200

    wandb.init(project="taxo-morph-pretrain", entity="michael-ginn", config={
        "bert-size": arch_size,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    })

    random.seed(13)

    train_data = load_data_file(f"../data/usp-train-track2-uncovered")
    dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

    print("Preparing datasets...")

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
    tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=MODEL_INPUT_LENGTH)

    dataset = DatasetDict()
    dataset['train'] = prepare_dataset_mlm(data=[line.morphemes() for line in train_data], tokenizer=tokenizer, device=device)
    dataset['dev'] = prepare_dataset_mlm(data=[line.morphemes() for line in dev_data], tokenizer=tokenizer, device=device)

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
        output_dir=f"../training-checkpoints",
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

    trainer.save_model(f'./usp-lang-model')

if __name__ == "__main__":
    train()