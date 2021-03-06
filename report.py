import csv
import os
import sys

import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    top_k_accuracy_score
)

from main import contexts

data_dir = sys.argv[1]
output_dir = sys.argv[2]

y_true = []
y_pred = []

y_score = []

with open(os.path.join(data_dir, 'test.csv'), newline='') as f:
    reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
        y_true.append(contexts.index(row['context']))

with open(os.path.join(output_dir, 'preds.csv'), newline='') as f:
    reader = csv.reader(f)
    y_score = np.array(list(reader), dtype=np.float64)

    y_pred = y_score.argmax(axis=-1)

print(classification_report(y_true, y_pred, digits=4))

print(f'accuracy\t{accuracy_score(y_true, y_pred)}')
for k in [5, 10]:
    print(f'top {k} accuracy\t{top_k_accuracy_score(y_true, y_score, k=k)}')
