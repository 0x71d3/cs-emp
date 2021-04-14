from torch import numel
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data import emotions, read_dd_split, EmotionDataset
from train import compute_metrics


def main():
    train_texts, train_labels = read_dd_split(
        'ijcnlp_dailydialog/train/dialogues_train.txt',
        'ijcnlp_dailydialog/train/dialogues_emotion_train.txt',
    )
    val_texts, val_labels = read_dd_split(
        'ijcnlp_dailydialog/validation/dialogues_validation.txt',
        'ijcnlp_dailydialog/validation/dialogues_emotion_validation.txt',
    )
    test_texts, test_labels = read_dd_split(
        'ijcnlp_dailydialog/test/dialogues_test.txt',
        'ijcnlp_dailydialog/test/dialogues_emotion_test.txt',
    )

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, add_special_tokens=False)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, add_special_tokens=False)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=False)

    del train_texts, val_texts, test_texts

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    del train_encodings, val_encodings, test_encodings
    del train_labels, val_labels, test_labels

    num_labels = len(emotions)
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results_dd',
        num_train_epochs=10,
        per_device_train_batch_size=4,  # 8 by default
        per_device_eval_batch_size=4,   # 8 by default
        gradient_accumulation_steps=8,  # 4 by default
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=0.0,
        warmup_steps=1634,
        logging_dir='./logs_dd',

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

    trainer.save_model('./roberta_dd')


if __name__ == '__main__':
    main()
