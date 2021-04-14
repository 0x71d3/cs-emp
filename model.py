import torch
import torch.nn as nn
from transformers import RobertaModel, BartModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaComet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.roberta = RobertaModel.from_pretrained('roberta-large', num_labels=num_labels)
        self.comet = BartModel.from_pretrained('comet-atomic_2020_BART')

        self.classification_head = BartClassificationHead(
            self.roberta.config.hidden_size + self.comet.config.d_model,
            self.roberta.config.hidden_size + self.comet.config.d_model,
            num_labels,
            self.roberta.config.hidden_dropout_prob,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        comet_input_ids=None,
        comet_attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        # RoBERTa
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]

        # COMET
        comet_outputs = self.comet(
            comet_input_ids,
            attention_mask=comet_attention_mask,
        )
        comet_hidden_states = comet_outputs[0]

        comet_eos_mask = comet_input_ids.eq(self.comet.config.eos_token_id)

        if len(torch.unique(comet_eos_mask.sum(1))) > 1:
            raise ValueError('All examples must have the same number of <eos> tokens.')
        comet_sentence_representation = comet_hidden_states[comet_eos_mask, :].view(comet_hidden_states.size(0), -1, comet_hidden_states.size(-1))[:, -1, :]

        classification_head_input = torch.cat(
            (
                sequence_output[:, 0, :],
                comet_sentence_representation,
            ),
            dim=-1,
        )
        logits = self.classification_head(classification_head_input)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaCometNoGrad(RobertaComet):
    def __init__(self, num_labels):
        super().__init__(num_labels)

        for param in self.comet.base_model.parameters():
            param.requires_grad = False
