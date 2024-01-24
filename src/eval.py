"""Contains the evaluation scripts for comparing predicted and gold IGT"""

from typing import List
# from data_handling import load_data_file
from torchtext.data.metrics import bleu_score
import click
import json
import evaluate
from transformers import EvalPrediction
from sklearn.metrics import f1_score


def compute_metrics(labels, hierarchy_matrix):
    def _compute_metrics(eval_preds: EvalPrediction):
        preds, gold_labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        print("PREDS", preds)
        print("LABELS", gold_labels)
        if len(gold_labels.shape) > 2:
            gold_labels = gold_labels.take(axis=1, indices=0)

        print(gold_labels.shape)

        def decode(pred_label_seq, gold_label_seq):
            decoded_labels = []
            decoded_preds = []
            decoded_label_categories = []
            decoded_pred_categories = []
            for pred_label, gold_label in zip(pred_label_seq, gold_label_seq):
                if gold_label == -100:
                    continue
                decoded_labels.append(labels[gold_label] if 0 <= gold_label < len(labels) else labels[0])
                decoded_preds.append(labels[pred_label] if 0 <= pred_label < len(labels) else labels[0])

                decoded_label_categories.append(
                    hierarchy_matrix[2][gold_label] if 0 <= gold_label < len(labels) else -1)
                decoded_pred_categories.append(hierarchy_matrix[2][pred_label] if 0 <= pred_label < len(labels) else -1)

            return decoded_preds, decoded_labels, decoded_pred_categories, decoded_label_categories

        print(preds.shape)
        decoded_preds, decoded_gold, decoded_pred_categories, decoded_label_categories = zip(
            *[decode(pred_seq, gold_seq) for pred_seq, gold_seq in zip(preds, gold_labels)])

        print('Preds:\t', decoded_preds[0])
        print('Labels:\t', decoded_gold[0])

        accuracy = eval_accuracy(decoded_preds, decoded_gold)

        # Calculate f1 between decoded_preds and decoded_labels
        flat_true_labels = [label for sublist in decoded_gold for label in sublist]
        flat_predicted_labels = [label for sublist in decoded_preds for label in sublist]

        # Compute F1 score
        f1 = f1_score(flat_true_labels, flat_predicted_labels, average='macro')

        # Compute accuracy at the second level of hierarchy_matrix
        def compute_list_of_lists_accuracy(true_labels, predicted_labels):
            correct = 0
            total = 0

            for t_list, p_list in zip(true_labels, predicted_labels):
                # Count matches in the overlapping parts
                correct += sum(t == p for t, p in zip(t_list, p_list))
                # Total is the length of the true list (since missing predictions are errors)
                total += len(t_list)

            return correct / total if total > 0 else 0

        flat_pred_categories = [label for sublist in decoded_pred_categories for label in sublist]
        flat_true_categories = [label for sublist in decoded_label_categories for label in sublist]

        print("PRED CATEGORIES", decoded_pred_categories[0])
        print("TRUE CATEGORIES", decoded_label_categories[0])

        # Compute accuracy between two lists
        category_accuracy = compute_list_of_lists_accuracy(decoded_label_categories, decoded_pred_categories)

        category_f1 = f1_score(flat_true_categories, flat_pred_categories, average='macro')

        return {
            "accuracy": accuracy,
            "f1": f1,
            "category_accuracy": category_accuracy,
            "category_f1": category_f1
        }

    return _compute_metrics


chrf = evaluate.load('chrf')


def eval_morpheme_glosses(
        pred_morphemes: List[List[str]], gold_morphemes: List[List[str]]
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(pred_morphemes, gold_morphemes)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])

    # chrf_score = chrf.compute(
    #     predictions=pred_morphemes, references=[[[morpheme] for morpheme in sequence] for sequence in gold_morphemes], word_order=2
    # )

    return {"morpheme_accuracy": morpheme_eval, "classes": class_eval, "bleu": bleu}


def eval_accuracy(pred: List[List[str]], gold: List[List[str]]) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be in the correct position in the list."""
    total_correct_predictions = 0
    total_tokens = 0
    summed_accuracies = 0

    for (entry_pred, entry_gold, i) in zip(pred, gold, range(len(gold))):
        entry_correct_predictions = 0

        entry_gold_len = len([token for token in entry_gold if token != '<sep>'])

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if token_index < len(entry_pred) and \
                    entry_pred[token_index] == entry_gold[token_index] and \
                    entry_gold[token_index] not in ['[UNK]', '<sep>']:
                entry_correct_predictions += 1

        entry_accuracy = (entry_correct_predictions / entry_gold_len)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_tokens += entry_gold_len

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    return {'average_accuracy': average_accuracy, 'accuracy': overall_accuracy}


def eval_stems_grams(pred: List[List[str]], gold: List[List[str]]) -> dict:
    perf = {'stem': {'correct': 0, 'pred': 0, 'gold': 0}, 'gram': {'correct': 0, 'pred': 0, 'gold': 0}}

    for (entry_pred, entry_gold) in zip(pred, gold):
        for token_index in range(len(entry_gold)):

            # We can determine if a token is a stem or gram by checking if it is all uppercase
            token_type = 'gram' if entry_gold[token_index].isupper() else 'stem'
            perf[token_type]['gold'] += 1

            if token_index < len(entry_pred):
                pred_token_type = 'gram' if entry_pred[token_index].isupper() else 'stem'
                perf[pred_token_type]['pred'] += 1

                if entry_pred[token_index] == entry_gold[token_index]:
                    # Correct prediction
                    perf[token_type]['correct'] += 1

    stem_perf = {'prec': 0 if perf['stem']['pred'] == 0 else perf['stem']['correct'] / perf['stem']['pred'],
                 'rec': 0 if perf['gram']['gold'] == 0 else perf['stem']['correct'] / perf['stem']['gold']}
    if (stem_perf['prec'] + stem_perf['rec']) == 0:
        stem_perf['f1'] = 0
    else:
        stem_perf['f1'] = 2 * (stem_perf['prec'] * stem_perf['rec']) / (stem_perf['prec'] + stem_perf['rec'])

    gram_perf = {'prec': 0 if perf['gram']['pred'] == 0 else perf['gram']['correct'] / perf['gram']['pred'],
                 'rec': 0 if perf['gram']['gold'] == 0 else perf['gram']['correct'] / perf['gram']['gold']}
    if (gram_perf['prec'] + gram_perf['rec']) == 0:
        gram_perf['f1'] = 0
    else:
        gram_perf['f1'] = 2 * (gram_perf['prec'] * gram_perf['rec']) / (gram_perf['prec'] + gram_perf['rec'])
    return {'stem': stem_perf, 'gram': gram_perf}


def eval_word_glosses(pred_words: List[List[str]], gold_words: List[List[str]]):
    """Evaluates the performance at the morpheme level"""
    word_eval = eval_accuracy(pred_words, gold_words)
    bleu = bleu_score(pred_words, [[line] for line in gold_words])
    return {'word_level': word_eval, 'bleu': bleu}

# @click.command()
# @click.option("--pred", help="File containing predicted IGT", type=click.Path(exists=True), required=True)
# @click.option("--gold", help="File containing gold-standard IGT", type=click.Path(exists=True), required=True)
# def evaluate_igt(pred: str, gold: str):
#     """Performs evaluation of a predicted IGT file"""
#
#     pred = load_data_file(pred)
#     gold = load_data_file(gold)
#
#     pred_words = [line.gloss_list() for line in pred]
#     gold_words = [line.gloss_list() for line in gold]
#     word_eval = eval_accuracy(pred_words, gold_words)
#
#     pred_morphemes = [line.gloss_list(segmented=True) for line in pred]
#     gold_morphemes = [line.gloss_list(segmented=True) for line in gold]
#
#     all_eval = {'word_level': word_eval,
#                 **eval_morpheme_glosses(pred_morphemes=pred_morphemes, gold_morphemes=gold_morphemes)}
#     print(json.dumps(all_eval, sort_keys=True, indent=4))
#
#
# if __name__ == '__main__':
#     evaluate_igt()
