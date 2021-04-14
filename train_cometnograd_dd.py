import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from data import emotions, read_comet_dd_split, CometEmotionDataset
from model import RobertaCometNoGrad
from train import compute_metrics
from train_cometnograd import CometTrainer


def main():
    train_texts, train_labels, comet_train_texts = read_comet_dd_split(
        'ijcnlp_dailydialog/train/dialogues_train.txt',
        'ijcnlp_dailydialog/train/dialogues_emotion_train.txt',
    )
    val_texts, val_labels, comet_val_texts = read_comet_dd_split(
        'ijcnlp_dailydialog/validation/dialogues_validation.txt',
        'ijcnlp_dailydialog/validation/dialogues_emotion_validation.txt',
    )
    test_texts, test_labels, comet_test_texts = read_comet_dd_split(
        'ijcnlp_dailydialog/test/dialogues_test.txt',
        'ijcnlp_dailydialog/test/dialogues_emotion_test.txt',
    )

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, add_special_tokens=False)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, add_special_tokens=False)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=False)

    comet_tokenizer = BartTokenizer.from_pretrained('comet-atomic_2020_BART')

    comet_train_encodings = comet_tokenizer(comet_train_texts, truncation=True, padding=True)
    comet_val_encodings = comet_tokenizer(comet_val_texts, truncation=True, padding=True)
    comet_test_encodings = comet_tokenizer(comet_test_texts, truncation=True, padding=True)

    del train_texts, val_texts, test_texts

    train_dataset = CometEmotionDataset(train_encodings, train_labels, comet_train_encodings)
    val_dataset = CometEmotionDataset(val_encodings, val_labels, comet_val_encodings)
    test_dataset = CometEmotionDataset(test_encodings, test_labels, comet_test_encodings)

    del train_encodings, val_encodings, test_encodings
    del train_labels, val_labels, test_labels
    del comet_train_encodings, comet_val_encodings, comet_test_encodings

    num_labels = len(emotions)
    model = RobertaCometNoGrad(num_labels)

    training_args = TrainingArguments(
        output_dir='./results_cometnograd_dd',
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=0.0,
        warmup_steps=1634,
        logging_dir='./logs_cometnograd_dd',

        evaluation_strategy='epoch',
        load_best_model_at_end=True,
    )

    early_stopping_callback = EarlyStoppingCallback()

    trainer = CometTrainer(
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

    trainer.save_model('./robertacometnograd_dd')


if __name__ == '__main__':
    main()
