import csv

import torch

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

emotions = [
    'no emotion', 'anger', 'disgust', 'fear',
    'happiness', 'sadness', 'surprise'
]


def read_ed_split(split_file):
    texts = []
    labels = []
    with open(split_file, newline='') as f:
        conv_id = ''
        utterances = []
        for row in csv.DictReader(f, quoting=csv.QUOTE_NONE):
            if conv_id and row['conv_id'] != conv_id:
                utterances = []
                
            conv_id = row['conv_id']
            utterances.append(row['utterance'].replace('_comma_', ','))
            if len(utterances) % 2 == 1:
                texts.append('<s>' + '</s></s>'.join(utterances) + '</s>')
                labels.append(contexts.index(row['context']))

    return texts, labels


def read_dd_split(text_file, label_file):
    texts = []
    labels = []
    with open(text_file) as f:
        for line in f:
            dialogue = list(
                map(lambda u: u.strip(), line.split('__eou__')[:-1])
            )
            for i in range(1, len(dialogue) + 1):
                texts.append('<s>' + '</s></s>'.join(dialogue[:i]) + '</s>')

    with open(label_file) as f:
        for line in f:
            labels += list(map(int, line.split()))

    assert len(texts) == len(labels)
    return texts, labels


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CometEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, comet_encodings):
        self.encodings = encodings
        self.labels = labels

        self.comet_encodings = comet_encodings

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])

        for key, val in self.comet_encodings.items():
            item['comet_' + key] = torch.tensor(val[idx])

        return item

    def __len__(self):
        return len(self.labels)
