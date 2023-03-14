from torch.utils.data import Dataset, DataLoader
from source_code.scripts.data_pre_processing import PreprocessData
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)


class FinalDatasets(Dataset):

    def __init__(self, dataset, model_tokenizer):
        self.dataset = dataset
        self.model_tokenizer = model_tokenizer
        self.data_dict = {}

    def get_data_dictionary(self):
        return self.data_dict

    def tokenize_and_encode_data(self, sentence):
        encoded_dict = self.model_tokenizer.encode_plus(

            sentence,
            add_special_tokens=False,

            # TODO: Need to discuss about the maximum tokenization length
            max_length=512,
            padding='max_length',
            # truncation=True,
            return_attention_mask=True,

            # return Pytorch Tensors.
            return_tensors='pt'
        )

        return encoded_dict

    def __getitem__(self, item):
        sentence = self.dataset[0][item]
        sentence_encoded_dict = self.tokenize_and_encode_data(sentence)
        token_list = sentence_encoded_dict['input_ids'][0]
        attention_mask = sentence_encoded_dict['attention_mask'][0]

        # labels = torch.tensor(self.dataset[1][item])
        label = self.dataset[1][item]
        label_encoded_dict = self.tokenize_and_encode_data(label)
        labels_list = label_encoded_dict['input_ids'][0]

        # entity_pairs = torch.tensor(self.dataset[2][item])
        entities = self.dataset[2][item]
        entities_encoded_dict = self.tokenize_and_encode_data(entities)
        entities_list = entities_encoded_dict['input_ids'][0]

        sample = (token_list, attention_mask, labels_list, entities_list)

        self.data_dict[item] = {'sentence': sentence,
                                'sentence_encoding': token_list,
                                'attention_mask': attention_mask,
                                'label': label,
                                'label_encoding': labels_list,
                                'entities': entities,
                                'entity_encoding': entities_list}

        return sample

    def __len__(self):
        return len(self.dataset[0])


class PreProcessedDataLoaders:

    def __init__(self, model_tokenizer):
        self.datasets = PreprocessData().initiate_data_processing()
        self.model_tokenizer = model_tokenizer

    def create_dataloader(self):

        training_dataset = FinalDatasets(self.datasets[0], self.model_tokenizer)
        validation_dataset = FinalDatasets(self.datasets[1], self.model_tokenizer)

        # TODO: Discuss about the batch size
        training_dataloader = DataLoader(training_dataset, batch_size=8)
        validation_dataloader = DataLoader(validation_dataset, batch_size=8)

        return training_dataloader, validation_dataloader
