from data import prepare_dataset, create_vocab, create_gloss_vocab, special_chars, load_data_file
import random
from tokenizer import WordLevelTokenizer
from uspanteko_morphology import morphology
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from taxonomic_loss_model import TaxonomicLossModel
import pandas as pd
import torch

# Load data
train_data = load_data_file(f"../data/usp-train-track2-uncovered")
dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=64)

glosses = create_gloss_vocab(morphology)

dataset = DatasetDict()
dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizer, labels=glosses, device='cpu')
dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizer, labels=glosses, device='cpu')


def compute_metrics(eval_preds):
    preds, gold_labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    total_morphemes = 0
    total_correct = 0
    total_top_5_correct = 0

    # Calculate top k accuracy
    for seq_index in range(len(gold_labels)):
        for token_index in range(len(gold_labels[seq_index])):
            correct_token_id = gold_labels[seq_index][token_index]
            if len(glosses) > correct_token_id >= 0 and correct_token_id != 1:
                total_morphemes += 1
                if correct_token_id in preds[seq_index][token_index]:
                    total_top_5_correct += 1
                if correct_token_id == preds[seq_index][token_index][0]:
                    total_correct += 1

    return {'accuracy': total_correct / total_morphemes, 'topkaccuracy': total_top_5_correct / total_morphemes}

def preprocess_logits_for_metrics(topk):
    def _preprocess_logits_for_metrics(logits, labels):
        return torch.topk(logits, topk, dim=2).indices
    return _preprocess_logits_for_metrics


args = TrainingArguments(
    output_dir=f"../training-checkpoints",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=3,
    num_train_epochs=100,
    load_best_model_at_end=True,
    report_to="wandb",
)

def _top_k_accuracy(size, k, seed):
    tax_model = TaxonomicLossModel.from_pretrained(f"./models/{size}-tax-{seed}-linear", num_labels=len(glosses))
    tax_model.use_morphology_tree(morphology, 5)

    harmonic_tax_model = TaxonomicLossModel.from_pretrained(f"./models/{size}-tax-{seed}-harmonic",
                                                            num_labels=len(glosses))
    harmonic_tax_model.use_morphology_tree(morphology, 5)

    flat_model = AutoModelForTokenClassification.from_pretrained(f"./models/{size}-flat-{seed}-linear",
                                                                 num_labels=len(glosses))

    tax_trainer = Trainer(
        tax_model,
        args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics(k),
    )

    harmonic_trainer = Trainer(
        harmonic_tax_model,
        args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics(k),
    )

    flat_trainer = Trainer(
        flat_model,
        args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics(k),
    )

    flat_acc = flat_trainer.evaluate(dataset['dev'])['eval_topkaccuracy']
    tax_acc = tax_trainer.evaluate(dataset['dev'])['eval_topkaccuracy']
    harmonic_acc = harmonic_trainer.evaluate(dataset['dev'])['eval_topkaccuracy']
    return (flat_acc, tax_acc, harmonic_acc)


def eval_topk(size, k):
    tax_accs = []
    harmonic_accs = []
    flat_accs = []

    for seed in range(42, 47):
        flat_acc, tax_acc, harmonic_acc = _top_k_accuracy(size, k, seed)

        flat_accs.append(flat_acc)
        tax_accs.append(tax_acc)
        harmonic_accs.append(harmonic_acc)

    return sum(flat_accs) / len(flat_accs), sum(tax_accs) / len(tax_accs), sum(harmonic_accs) / len(harmonic_accs)

all_results = []

for size in [10, 100, 500, 1000, 'full']:
    size_results = []
    for k in [1, 2, 3, 4, 5, 6]:
        flat_score, tax_score, harmonic_score = eval_topk(size, k) if size != 'full' else _top_k_accuracy(size, k, 1)
        size_results.append((flat_score, tax_score, harmonic_score))
    all_results.append(size_results)

df = pd.DataFrame(all_results)
df.applymap(lambda x: (round(x[0], 3), round(x[1], 3), round(x[2], 3)))
df.to_excel("topk.xlsx")