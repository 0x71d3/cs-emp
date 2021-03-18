import argparse
import csv
import os
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

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


class EmotionDataset(Dataset):
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


class LitRoberta(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        config = RobertaConfig.from_pretrained(
            'roberta-large',
            num_labels=self.hparams.num_labels
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-large',
            config=config
        )

        # loader
        dataset = EmotionDataset(
            data_dir=self.hparams.data_dir,
            split='train',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True
        )

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        self.log('test_loss', loss, prog_bar=True)

        logits = outputs.logits
        return logits.softmax(dim=-1)

    def test_epoch_end(self, outputs):
        preds = torch.cat(outputs)

        with open(
            os.path.join(self.hparams.output_dir, 'preds.csv'),
            'w',
            newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerows(preds.tolist())

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
            lr=self.hparams.lr
        )

        # scheduler
        num_training_steps = (
            len(self.train_loader)
            // self.hparams.accumulate_grad_batches
            * self.hparams.max_epochs
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=num_training_steps
        )
        lr_dict = {'scheduler': lr_scheduler, 'interval': 'step'}

        return [optimizer], [lr_dict]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        dataset = EmotionDataset(
            data_dir=self.hparams.data_dir,
            split='valid',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.valid_batch_size
        )
        return loader

    def test_dataloader(self):
        dataset = EmotionDataset(
            data_dir=self.hparams.data_dir,
            split='test',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.valid_batch_size
        )
        return loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir')
    parser.add_argument('output_dir')

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.98, type=float)
    parser.add_argument('--adam_eps', default=1e-6, type=float)
    parser.add_argument('--num_warmup_steps', default=823, type=int)

    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--valid_batch_size', default=8, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--num_labels', default=32, type=int)

    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--gradient_clip_val', default=0.0, type=float)
    parser.add_argument('--max_epochs', default=10, type=int)

    args = parser.parse_args()
    
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=args.output_dir
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=1,
        mode='min'
    )
    
    trainer = Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True,
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        max_epochs=args.max_epochs
    )

    model = LitRoberta(args)

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
