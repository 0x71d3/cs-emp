import csv
import os

import torch
from torch.utils.data import Dataset

emotions = [
    'no emotion', 'anger', 'disgust', 'fear',
    'happiness', 'sadness', 'surprise'
]

contexts = [
    'afraid', 'angry', 'annoyed', 'anticipating',
    'anxious', 'apprehensive', 'ashamed', 'caring',
    'confident', 'content', 'devastated', 'disappointed',
    'disgusted', 'embarrassed', 'excited', 'faithful',
    'furious', 'grateful', 'guilty', 'hopeful',
    'impressed', 'jealous', 'joyful', 'lonely',
    'nostalgic', 'prepared', 'proud', 'sad',
    'sentimental', 'surprised', 'terrified', 'trusting'
]


class EmotionDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        texts = []
        labels = []

        self.inputs = tokenizer(
            texts, 
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.inputs['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.inputs['input_ids'][index]
        attention_mask = self.inputs['attention_mask'][index]
        labels = self.labels[index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DailyDialogEmotionDataset(EmotionDataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        if split == 'valid':
            split = 'validation'

        texts = []
        with open(
            os.path.join(data_dir, split, f'dialogues_{split}.txt')
        ) as f:
            for line in f:
                dialogue = list(
                    map(lambda u: u.strip(), line.split('__eou__')[:-1])
                )
                for i in range(1, len(dialogue) + 1):
                    texts.append(
                        tokenizer.cls_token
                        + (tokenizer.sep_token * 2).join(dialogue[:i])
                        + tokenizer.sep_token
                    )

        labels = []
        with open(
            os.path.join(data_dir, split, f'dialogues_emotion_{split}.txt'),
        ) as f:
            for line in f:
                labels += list(map(int, line.split()))

        assert len(texts) == len(labels)

        self.inputs = tokenizer(
            texts, 
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
        

class EmpatheticDialoguesEmotionDataset(EmotionDataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        texts = []
        labels = []

        with open(os.path.join(data_dir, split + '.csv'), newline='') as f:
            conv_id = ''
            utterances = []

            for row in csv.DictReader(f, quoting=csv.QUOTE_NONE):
                if conv_id and row['conv_id'] != conv_id:
                    utterances = []
                    
                conv_id = row['conv_id']
                utterances.append(row['utterance'].replace('_comma_', ','))

                if len(utterances) % 2 == 1:
                    texts.append(
                        tokenizer.cls_token
                        + (tokenizer.sep_token * 2).join(utterances)
                        + tokenizer.sep_token
                    )
                    labels.append(contexts.index(row['context']))

        self.inputs = tokenizer(
            texts, 
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
