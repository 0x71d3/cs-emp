import csv
import os
import sys

import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    top_k_accuracy_score
)

from main import contexts, emotions

data_dir = sys.argv[1]
output_dir = sys.argv[2]

y_true = []
y_pred = []

y_score = []

if 'empatheticdialogues' in data_dir:
    with open(os.path.join(data_dir, 'test.csv'), newline='') as f:
        conv_id = ''
        num_utterances = 0

        reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in reader:
            if conv_id != row['conv_id']:
                num_utterances = 0
            
            num_utterances += 1
            if num_utterances % 2 == 1:
                y_true.append(contexts.index(row['context']))

            conv_id = row['conv_id']
    
    target_names = contexts

else:  # DailyDialog
    with open(os.path.join(data_dir, 'test', 'dialogues_emotion_test.txt')) as f:
        for line in f:
            y_true += list(map(int, line.split()))

    target_names = emotions

with open(os.path.join(output_dir, 'preds.csv'), newline='') as f:
    reader = csv.reader(f)
    y_score = np.array(list(reader), dtype=np.float64)

    y_pred = y_score.argmax(axis=-1)

print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

print(f'accuracy\t{accuracy_score(y_true, y_pred)}\n')
for k in range(2, len(target_names)):
    print(f'top {k} accuracy\t{top_k_accuracy_score(y_true, y_score, k=k)}')
