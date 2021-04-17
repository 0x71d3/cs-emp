import argparse
from re import L

import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data import (
    contexts,
    emotions,
    read_ed_split,
    read_dd_split,
    read_comet_ed_split,
    read_comet_dd_split,
    EmotionDataset,
    CometEmotionDataset,
)
from model import RobertaComet, RobertaCometNoGrad


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


class CometTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main(args):
    if 'empatheticdialogues' in args.data_dir:
        num_labels = len(contexts)

        if not args.comet:
            train_texts, train_labels = read_ed_split(f'{args.data_dir}/train.csv')
            val_texts, val_labels = read_ed_split(f'{args.data_dir}/valid.csv')
            test_texts, test_labels = read_ed_split(f'{args.data_dir}/test.csv')

        else:
            train_texts, train_labels, comet_train_texts = read_comet_ed_split(f'{args.data_dir}/train.csv')
            val_texts, val_labels, comet_val_texts = read_comet_ed_split(f'{args.data_dir}/valid.csv')
            test_texts, test_labels, comet_test_texts = read_comet_ed_split(f'{args.data_dir}/test.csv')

    else:
        num_labels = len(emotions)

        if not args.comet:
            train_texts, train_labels = read_dd_split(
                f'{args.data_dir}/train/dialogues_train.txt',
                f'{args.data_dir}/train/dialogues_emotion_train.txt',
            )
            val_texts, val_labels = read_dd_split(
                f'{args.data_dir}/validation/dialogues_validation.txt',
                f'{args.data_dir}/validation/dialogues_emotion_validation.txt',
            )
            test_texts, test_labels = read_dd_split(
                f'{args.data_dir}/test/dialogues_test.txt',
                f'{args.data_dir}/test/dialogues_emotion_test.txt',
            )

        else:
            train_texts, train_labels, comet_train_texts = read_comet_dd_split(
                f'{args.data_dir}/train/dialogues_train.txt',
                f'{args.data_dir}/train/dialogues_emotion_train.txt',
            )
            val_texts, val_labels, comet_val_texts = read_comet_dd_split(
                f'{args.data_dir}/validation/dialogues_validation.txt',
                f'{args.data_dir}/validation/dialogues_emotion_validation.txt',
            )
            test_texts, test_labels, comet_test_texts = read_comet_dd_split(
                f'{args.data_dir}/test/dialogues_test.txt',
                f'{args.data_dir}/test/dialogues_emotion_test.txt',
            )

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_name_or_path)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, add_special_tokens=False)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, add_special_tokens=False)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=False)

    if args.comet:
        comet_tokenizer = BartTokenizer.from_pretrained('comet-atomic_2020_BART')

        comet_train_encodings = comet_tokenizer(comet_train_texts, truncation=True, padding=True)
        comet_val_encodings = comet_tokenizer(comet_val_texts, truncation=True, padding=True)
        comet_test_encodings = comet_tokenizer(comet_test_texts, truncation=True, padding=True)

    del train_texts, val_texts, test_texts

    if not args.comet:
        train_dataset = EmotionDataset(train_encodings, train_labels)
        val_dataset = EmotionDataset(val_encodings, val_labels)
        test_dataset = EmotionDataset(test_encodings, test_labels)
    
    else:
        train_dataset = CometEmotionDataset(train_encodings, train_labels, comet_train_encodings)
        val_dataset = CometEmotionDataset(val_encodings, val_labels, comet_val_encodings)
        test_dataset = CometEmotionDataset(test_encodings, test_labels, comet_test_encodings)

        del comet_train_encodings, comet_val_encodings, comet_test_encodings

    del train_encodings, val_encodings, test_encodings
    del train_labels, val_labels, test_labels

    if not args.comet:
        model = RobertaForSequenceClassification.from_pretrained(args.roberta_name_or_path, num_labels=num_labels)
    
    else:
        if not args.comet_no_grad:
            model = RobertaComet(num_labels)
        
        else:
            model = RobertaCometNoGrad(num_labels)

    model_name = 'rb-'
    if 'base' in args.roberta_name_or_path:
        model_name += 'b'
    else:
        model_name += 'l'
    if args.comet:
        model_name += '_c-'
        if 'base' in args.comet_name_or_path:
            model_name += 'b'
        else:
            model_name += 'l'
        if args.comet_no_grad:
            model_name += '-ng'
    model_name += '_'
    if 'empatheticdialogues' in args.data_dir:
        model_name += 'ed'
    else:
        model_name += 'dd'

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',
        num_train_epochs=10,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=0.0,
        warmup_steps=823,
        logging_dir=f'./logs/{model_name}',

        evaluation_strategy='epoch',
        load_best_model_at_end=True,
    )

    early_stopping_callback = EarlyStoppingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    predict_output = trainer.predict(test_dataset)
    print(predict_output.metrics)

    trainer.save_model(f'./models/{model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='empatheticdialogues')

    parser.add_argument('--roberta_name_or_path', default='roberta-large')

    parser.add_argument('--comet', action='store_true')
    parser.add_argument('--comet_no_grad', action='store_true')
    parser.add_argument('--comet_name_or_path', default='comet-atomic_2020_BART')

    parser.add_argument('--per_device_batch_size', default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)

    args = parser.parse_args()
    main(args)
