from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaForTokenClassification, BertConfig, XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer
from source_code.scripts.dataloaders import PreProcessedDataLoaders
import torch
import logging
from torch import nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)


class XLMRobertaPretrained:

    # Refer https://github.com/huggingface/tokenizers/issues/247 for more information regarding what special tokens
    # are meant to do and how fine-tuning the model can help create better embeddings with these newly added tokens
    def add_special_tokens(self):
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def __init__(self, model_type):
        # Initializing the pre-trained model. This does not initialize the pre-trained weights but only the
        # configurations.
        config = XLMRobertaConfig.from_pretrained(model_type)
        self.model = XLMRobertaModel.from_pretrained(model_type, config=config)

        # Initializing the pre-trained tokenizer. This tokenizer uses Sub-word Tokenization technique. XLMRoberta uses
        # uni-gram based SentencePiece encoding scheme to encode the data into vector embeddings. For more
        # information, refer to https://huggingface.co/transformers/tokenizer_summary.html.
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_type)
        self.add_special_tokens()

        # Fetching the dataloader for both training and validation data, created using the pre-processed and
        # pre-tokenized sentences in the given dataset. Refer to https://www.kaggle.com/c/indore-datathon-2021 for
        # more details about the data.
        self.training_dataloader, self.validation_dataloader = PreProcessedDataLoaders(self.tokenizer).create_dataloader()

        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        seed_val = 123
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Moving the model to 'cpu' or 'gpu' depending on the available device resources.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def retrieve_vector_embeddings(self):
        token_outputs, label_outputs, entity_outputs = [], [], []
        for token_list, attention_mask, labels_list, entities_list in iter(self.training_dataloader):
            # print(token_list.shape)
            # print(attention_mask.shape)
            # print(labels_list.shape)
            # print(entities_list.shape)

            temp_token_outputs = self.model(token_list, attention_mask=attention_mask)
            temp_label_outputs = self.model(labels_list)
            temp_entity_outputs = self.model(entities_list)

            token_outputs.append(temp_token_outputs.last_hidden_state)
            label_outputs.append(temp_label_outputs.last_hidden_state)
            entity_outputs.append(temp_entity_outputs.last_hidden_state)

            # print(token_outputs)
            # print(label_outputs)
            # print(entity_outputs)

        tokens = torch.cat(token_outputs, dim=0) # Hello World -> 1223, 1234
        labels = torch.cat(label_outputs, dim=0) # Organization -> 45765785
        entities = torch.cat(entity_outputs, dim=0) # [E1, E2] -> 476487658

        return token_outputs, label_outputs, entity_outputs

    def model_fine_tuning(self):

        optimizer = AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)
        epochs = 5
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(self.training_dataloader)*epochs)

        for epoch in tqdm(range(1, epochs+1)):
            self.model.train()
            loss_train_total = 0

            progress_bar = tqdm(self.training_dataloader,
                                desc='Epoch {:1d}'.format(epoch),
                                leave=False,
                                disable=False)

            for batch in progress_bar:
                self.model.zero_grad()
        #         batch = tuple(b.to(self.device) for b in batch)
        #         inputs = {
        #             'input_ids':        batch[0],
        #             'attention_mask':   batch[1],
        #             'labels':           batch[2],
        #         }
        #
        #         outputs = self.model(**inputs)
        #         loss = outputs[0]
        #         loss_train_total += loss.item()
        #         loss.backward()
        #
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        #
        #         optimizer.step()
        #         scheduler.step()
        #
        #         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        #
        #     tqdm.write(f'\nEpoch {epoch}')
        #
        #     loss_train_avg = loss_train_total/len(self.training_dataloader)
        #     tqdm.write(f'Training loss: {loss_train_avg}')
        #
        #     # val_loss, predictions, true_vals = evaluate(dataloader_val)
        #     # val_f1 = f1_score_func(predictions, true_vals)
        #     # tqdm.write(f'Validation loss: {val_loss}')
        #     # tqdm.write(f'F1 Score (weighted): {val_f1}')
        #
        # torch.save(self.model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/DL4NLP/')

        for token_list, attention_mask, labels_list, entities_list in iter(self.training_dataloader):
            token_list = token_list.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels_list = labels_list.to(self.device)
            entities_list = entities_list.to(self.device)

            outputs = self.model(token_list, attention_mask)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            _, predictions = torch.max(outputs, dim=1)
            labels = torch.max(labels_list, dim=1)
            loss = self.loss_function(outputs, labels)
            loss.backward()

            # TODO: Add Optimizer and Scheduler


if __name__ == '__main__':
    # Pre-trained model types depending on the vocabulary size. We do not use the entire list of pre-trained
    # configuration. For more information, refer to the following link:
    # https://huggingface.co/transformers/_modules/transformers/models/xlm_roberta/tokenization_xlm_roberta.html#XLMRobertaTokenizer
    MODEL_TYPES = ['xlm-roberta-base', 'xlm-roberta-large']
    xlm_roberta_pretrained = XLMRobertaPretrained(MODEL_TYPES[0])
    token_embeddings, label_embeddings, entity_embeddings = xlm_roberta_pretrained.retrieve_vector_embeddings()
