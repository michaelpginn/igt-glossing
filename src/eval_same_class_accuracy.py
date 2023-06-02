"""Evaluates predictions made on unseen morphemes with seen glosses"""
import random
import torch
import pandas as pd
from datasets import DatasetDict
from transformers import AutoModelForTokenClassification

from data import prepare_dataset, create_vocab, create_gloss_vocab, special_chars, load_data_file
from tokenizer import WordLevelTokenizer
from uspanteko_morphology import morphology
from finetune_token_classifier import create_trainer


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def eval_model(seed, all_glosses, train_data, dataset, tokenizer):
    # Create a set of the morphemes and glosses that were seen in the training data
    random.seed(seed)
    train_data = random.sample(train_data, 10)

    train_seen_morphemes = set()
    train_seen_glosses = set()

    for row in train_data:
        for morpheme in row.morphemes():
            train_seen_morphemes.add(morpheme)
        for label in row.gloss_list(segmented=True):
            train_seen_glosses.add(label)

    print(f"Train data includes {len(train_seen_glosses)}/{len(all_glosses)} of possible labels")

    # Run prediction on the dev set
    model = AutoModelForTokenClassification.from_pretrained(f"../../models/10-flat-{seed}-linear", num_labels=len(all_glosses))
    trainer = create_trainer(model, dataset=dataset, tokenizer=tokenizer, labels=all_glosses, batch_size=64, max_epochs=30)
    preds = trainer.predict(dataset['dev'])

    # Evaluate correct predictions on unseen morphemes
    # How was performance on unseen morphemes with seen glosses?
    count_seen_morpheme = 0
    count_unseen_morpheme_seen_gloss = 0
    count_unseen_morpheme_unseen_gloss = 0
    count_unseen_morpheme_seen_gloss_correct = 0
    count_unseen_morpheme_unseen_gloss_correct = 0

    for row_index in range(len(dataset['dev'])):
        row = dataset['dev'][row_index]
        pred_row = preds[0][row_index]

        for token_index in range(len(row['morphemes'])):
            morpheme = row['morphemes'][token_index]
            correct_label_index = row['labels'][token_index]
            predicted_label_index = pred_row[token_index]

            # Skip PAD
            if correct_label_index == 1:
                continue

            # Skip seen morphemes
            if morpheme in train_seen_morphemes:
                count_seen_morpheme += 1
                continue

            # Check if gloss was seen
            if all_glosses[correct_label_index] in train_seen_glosses:
                count_unseen_morpheme_seen_gloss += 1
                if predicted_label_index == correct_label_index:
                    count_unseen_morpheme_seen_gloss_correct += 1
            else:
                count_unseen_morpheme_unseen_gloss += 1
                if predicted_label_index == correct_label_index:
                    count_unseen_morpheme_unseen_gloss_correct += 1

    return {"seen_gloss_acc": count_unseen_morpheme_seen_gloss_correct / count_unseen_morpheme_seen_gloss,
            "unseen_gloss_acc": count_unseen_morpheme_unseen_gloss_correct / count_unseen_morpheme_unseen_gloss,
            "total_unseen_morpheme_seen_gloss": count_unseen_morpheme_seen_gloss,
            "total_unseen_morpheme_unseen_gloss": count_unseen_morpheme_unseen_gloss,
            "total_unseen_morpheme": count_unseen_morpheme_seen_gloss + count_unseen_morpheme_unseen_gloss,
            "total_morphemes": count_unseen_morpheme_seen_gloss + count_unseen_morpheme_unseen_gloss + count_seen_morpheme}


def main():
    # Load data
    train_data = load_data_file(f"../data/usp-train-track2-uncovered")
    dev_data = load_data_file(f"../data/usp-dev-track2-uncovered")

    train_vocab = create_vocab([line.morphemes() for line in train_data], threshold=1)
    tokenizer = WordLevelTokenizer(vocab=train_vocab, model_max_length=64)

    all_glosses = create_gloss_vocab(morphology)

    dataset = DatasetDict()
    dataset['train'] = prepare_dataset(data=train_data, tokenizer=tokenizer, labels=all_glosses, device=device)
    dataset['dev'] = prepare_dataset(data=dev_data, tokenizer=tokenizer, labels=all_glosses, device=device)

    all_results = []
    for seed in range(1, 21):
        all_results.append(eval_model(seed, all_glosses, train_data, dataset, tokenizer))

    df = pd.DataFrame(all_results)
    df.to_csv('same_class_accuracy.csv')


if __name__ == '__main__':
    main()