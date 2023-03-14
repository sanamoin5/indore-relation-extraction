# Importing python libraries
import csv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)


class PreprocessData:

    def __init__(self):
        self.dataset_type = ["training", "validation"]

    # Opening the respective datafiles to get the respective data
    def __fetch_data(self, file_name, dataset_type):
        batch_sentences = []
        batch_relations = []
        batch_entity_pair = []

        with open(file_name) as data_file:
            data_reader = csv.reader(data_file, delimiter="\t")
            next(data_reader)
            for line in data_reader:

                if dataset_type == self.dataset_type[0]:
                    batch_sentences.append("".join(['<s>', line[1], '</s>']))
                    batch_relations.append(line[0])
                    batch_entity_pair.append(line[2:])
                else:
                    batch_sentences.append(line[1])

        if dataset_type == self.dataset_type[0]:
            unique_labels = list(set(batch_relations))
            label_dict = {label: index for index, label in enumerate(unique_labels)}

            return batch_sentences, batch_relations, batch_entity_pair, label_dict
        else:
            return batch_sentences, batch_relations, batch_entity_pair

    def initiate_data_processing(self):
        training_dataset = self.__fetch_data("../../data/train.tsv", self.dataset_type[0])
        validation_dataset = self.__fetch_data("../../data/valid.tsv", self.dataset_type[1])

        return training_dataset, validation_dataset
