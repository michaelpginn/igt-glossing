"""Defines models and functions for loading, manipulating, and writing task data"""
from typing import Optional, List
import re
import random
from functools import reduce
from datasets import Dataset, DatasetDict
import torch
from encoder import create_vocab, CustomEncoder
from custom_tokenizers import morpheme_tokenize_no_punc as tokenizer

class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: Optional[str], translation: Optional[str]):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation
        self.should_segment = True

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return []
        if not segmented:
            return self.glosses.split()
        else:
            words = re.split("\s+", self.glosses)
            glosses = [re.split("-", word) for word in words]
            glosses = [[gloss for gloss in word_glosses if gloss != ''] for word_glosses in glosses] # Remove empty glosses introduced by faulty segmentation
            glosses = [word_glosses for word_glosses in glosses if word_glosses != []]
            glosses = reduce(lambda a,b: a + ['[SEP]'] + b, glosses) # Add separator for word boundaries
            return glosses

    def morphemes(self) -> Optional[List[str]]:
        """Returns the segmented list of morphemes, if possible"""
        if self.segmentation is None:
            return None
        return tokenizer(self.segmentation)


    def __dict__(self):
        d = {'transcription': self.transcription, 'translation': self.translation}
        if self.glosses is not None:
            d['glosses'] = self.gloss_list(segmented=self.should_segment)
        if self.segmentation is not None:
            d['segmentation'] = self.segmentation
            d['morphemes'] = self.morphemes()
        return d


def load_data_file(path: str) -> List[IGTLine]:
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\p' and current_entry[2] == None:
                if len(line[3:].strip()) > 0:
                    current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line.strip() != "":
                # Something went wrong
                continue
            else:
                if not current_entry == [None, None, None, None]:
                    all_data.append(IGTLine(transcription=current_entry[0],
                                            segmentation=current_entry[1],
                                            glosses=current_entry[2],
                                            translation=None))
                    current_entry = [None, None, None, None]
        # Might have one extra line at the end
        if not current_entry == [None, None, None, None]:
            all_data.append(IGTLine(transcription=current_entry[0],
                                    segmentation=current_entry[1],
                                    glosses=current_entry[2],
                                    translation=None))
    return all_data


def create_gloss_vocab(morphology):
    def parse_tree(morphology_subtree):
        all_glosses = []
        for item in morphology_subtree:
            if isinstance(item, tuple):
                all_glosses += parse_tree(item[1])
            else:
                all_glosses.append(item)
        return all_glosses

    return parse_tree(morphology)


def prepare_dataset_mlm(data: List[List[str]], encoder: CustomEncoder, model_input_length: int, mlm_probability: float, device):
    """Encodes, pads, and creates attention mask for input. Also masks tokens and sets labels according"""

    # Create a dataset
    raw_dataset = Dataset.from_list([{'strs': line} for line in data])

    def process(row):
        source_enc = encoder.encode(row['strs'], vocab='input')
        masked_source = []
        labels = []

        # Mask tokens
        for token in source_enc:
            if random.random() < mlm_probability:
                # Mask the token
                mask_type_prob = random.random()

                # 80% use MASK, 10% use random token, 10% unchanged
                if mask_type_prob < 0.8:
                    masked_source.append(encoder.MASK_ID)
                elif mask_type_prob < 0.9:
                    masked_source.append(encoder.random_token_id())
                else:
                    masked_source.append(token)

                labels.append(token)
            else:
                masked_source.append(token)

                # Ignore non-masked tokens
                labels.append(-100)

        print(masked_source)

        # Pad
        initial_length = len(masked_source)
        masked_source += [encoder.PAD_ID] * (model_input_length - initial_length)
        labels += [-100] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (model_input_length - initial_length)

        # Encode the output, if present
        return { 'input_ids': torch.tensor(masked_source, dtype=torch.long).to(device),
                 'attention_mask': torch.tensor(attention_mask).to(device),
                 'labels': torch.tensor(labels, dtype=torch.long).to(device)}

    return raw_dataset.map(process)


def prepare_dataset(data: List[IGTLine], encoder: CustomEncoder, model_input_length: int, device):
    """Encodes and pads inputs and creates attention mask"""

    # Create a dataset
    raw_dataset = Dataset.from_list([line.__dict__() for line in data])

    def process(row):
        source_enc = encoder.encode(row['morphemes'], vocab='input')

        # Pad
        initial_length = len(source_enc)
        source_enc += [encoder.PAD_ID] * (model_input_length - initial_length)

        # Create attention mask
        attention_mask = [1] * initial_length + [0] * (model_input_length - initial_length)

        # Encode the output, if present
        if 'glosses' in row:
            # For token class., the labels are just the glosses for each word
            output_enc = encoder.encode(row['glosses'], vocab='output')
            output_enc += [-100] * (model_input_length - len(output_enc))
            return { 'input_ids': torch.tensor(source_enc, dtype=torch.long).to(device),
                     'attention_mask': torch.tensor(attention_mask).to(device),
                     'labels': torch.tensor(output_enc, dtype=torch.long).to(device)}

        else:
            # If we have no glosses, this must be a prediction task or language modeling
            return { 'input_ids': torch.tensor(source_enc).to(device),
                     'attention_mask': torch.tensor(attention_mask).to(device)}

    return raw_dataset.map(process)


def write_predictions(path: str, preds, pred_input_data, encoder: CustomEncoder, from_vocabulary_index=None):
    """Writes the predictions to a new file, which uses the file in `path` as input"""
    def create_gloss_line(glosses, transcription_tokens):
        """
        Write a gloss for each transcription token
        We should never write more glosses than there are tokens
        If tokens are segmented, write morphemes together
        """
        output_line = ''
        for (token, gloss) in zip(transcription_tokens, glosses):
            if token[0] == '-':
                output_line += f"-{gloss}"
            else:
                output_line += f" {gloss}"
        return output_line

    decoded_preds = encoder.batch_decode(preds)
    next_line = 0
    with open(path, 'r') as input:
        with open('usp_output_preds', 'w') as output:
            for line in input:
                line_prefix = line[:2]
                if line_prefix == '\\g':
                    output_line = create_gloss_line(glosses=decoded_preds[next_line], transcription_tokens=pred_input_data[next_line]['tokenized_transcription'])
                    output_line = line_prefix + output_line + '\n'
                    output.write(output_line)
                    next_line += 1
                else:
                    output.write(line)
    print(f"Predictions written to ./usp_output_preds")
