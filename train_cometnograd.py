import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from data import read_comet_ed_split, CometEmotionDataset
from model import RobertaCometNoGrad
from train import compute_metrics


class CometTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


train_texts, train_labels, comet_train_texts = read_comet_ed_split('empatheticdialogues/train.csv')
val_texts, val_labels, comet_val_texts = read_comet_ed_split('empatheticdialogues/valid.csv')
test_texts, test_labels, comet_test_texts = read_comet_ed_split('empatheticdialogues/test.csv')

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

model = RobertaCometNoGrad.from_pretrained('roberta-large', num_labels=32)

training_args = TrainingArguments(
    output_dir='./results_cometnograd',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    warmup_steps=823,
    logging_dir='./logs_cometnograd',

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

trainer.save_model('./robertacometnograd_ed')
