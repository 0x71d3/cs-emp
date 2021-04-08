from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from data import read_ed_split, EmotionDataset


def main():
    train_texts, train_labels = read_ed_split('empatheticdialogues/train.csv')
    val_texts, val_labels = read_ed_split('empatheticdialogues/valid.csv')
    test_texts, test_labels = read_ed_split('empatheticdialogues/test.csv')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, add_special_tokens=False)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, add_special_tokens=False)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=False)

    del train_texts, val_texts, test_texts

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    del train_labels, val_labels, test_labels
    del train_encodings, val_encodings, test_encodings

    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=32)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=4,  # 8
        per_device_eval_batch_size=4,   # 8
        gradient_accumulation_steps=8,  # 4
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=0.0,
        warmup_steps=823,
        logging_dir='./logs',

        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='no',
        load_best_model_at_end=True
    )

    early_stopping_callback = EarlyStoppingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        callbacks=[early_stopping_callback]
    )

    trainer.train()

    trainer.save_model('./roberta_ed')


if __name__ == '__main__':
    main()
