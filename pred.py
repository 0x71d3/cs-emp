from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from sklearn.metrics import (
    classification_report,
    top_k_accuracy_score,
)

from data import contexts, read_ed_split, EmotionDataset


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_texts, test_labels = read_ed_split('empatheticdialogues/test.csv')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=False)
    del test_texts

    test_dataset = EmotionDataset(test_encodings, test_labels)
    del test_encodings, test_labels

    model = RobertaForSequenceClassification.from_pretrained('roberta_ed', num_labels=32)
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32)

    y_true = []
    y_pred = []
    y_score = []

    for batch in tqdm(test_loader):
        labels = batch['labels'].to(device)
        y_true.extend(labels.tolist())

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        logits = outputs.logits

        probs = F.softmax(logits, -1)
        y_score.extend(probs.tolist())

        preds = probs.argmax(-1)
        y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=contexts, digits=4))

    for k in range(1, len(contexts)):    
        print(f'top {k} accuracy score\t{top_k_accuracy_score(y_true, y_score, k=k)}')


if __name__ == '__main__':
    main()
