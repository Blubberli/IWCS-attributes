import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from data.feature_extractor import StaticEmbeddingExtractor
#from data.feature_extractor_contextualized import BertExtractor
from data.feature_extractor_contextualized_all_layers import BertExtractor
import numpy as np
import torch.nn.functional as F
import torch


def extract_all_labels(training_data, validation_data, test_data, separator, label):
    """
    This method returns a list of all unique labels that occur in a dataset
    :param training_data: the training data with a column named 'label'
    :param validation_data: the validation data with a column named 'label'
    :param test_data: the test data with a column named 'label'
    :param separator: the separator of the column based data
    :param label: the column name that stores the labels
    :return: a list with all unique labels of the dataset
    """
    training_labels = set(pd.read_csv(training_data, delimiter=separator, index_col=False)[label])
    validation_labels = set(pd.read_csv(validation_data, delimiter=separator, index_col=False)[label])
    test_labels = set(pd.read_csv(test_data, delimiter=separator, index_col=False)[label])
    all_labels = list(training_labels.union(validation_labels).union(test_labels))
    return all_labels


def create_label_encoder(all_labels):
    """
    This method creates a label encoder that encodes each label to a unique number
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder


class RankingDataset(Dataset):

    def __init__(self, data_path, general_embedding_path, label_embedding_path, separator, mod, head, label):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param general_embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param label_embedding_path: the path to the pretrained static word embeddings to lookup the label
        represenations
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels = list(self.data[label])
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=general_embedding_path)
        self._label_extractor = StaticEmbeddingExtractor(label_embedding_path)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        label_embeddings = self.label_extractor.get_array_embeddings(self.labels)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "label_rep": label_embeddings[i], "label": self.labels[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels(self):
        return self._labels

    @property
    def status(self):
        return self._status

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_extractor(self):
        return self._label_extractor

    @property
    def samples(self):
        return self._samples


class JointRankingDataset(Dataset):

    def __init__(self, data_path, general_embedding_path, label_embedding_path, separator, mod, head, label_1, label_2):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param general_embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param label_embedding_path: the path to the pretrained static word embeddings to lookup the label
        represenations
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels_1 = list(self.data[label_1])
        self._labels_2 = list(self.data[label_2])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels_1) == len(self.labels_2), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=general_embedding_path)
        self._label_extractor = StaticEmbeddingExtractor(label_embedding_path)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        attribute_embeddings = self.label_extractor.get_array_embeddings(self.labels_1)
        semclass_embeddings = self.label_extractor.get_array_embeddings(self.labels_2)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "attribute_rep": attribute_embeddings[i], "attribute": self.labels_1[i],
             "semclass_rep": semclass_embeddings[i], "semclass": self.labels_2[i]}
            for i in range(len(self.labels_1))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels_1(self):
        return self._labels_1

    @property
    def labels_2(self):
        return self._labels_2

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_extractor(self):
        return self._label_extractor

    @property
    def samples(self):
        return self._samples


class JointClassificationDataset(Dataset):

    def __init__(self, data_path, embedding_path, separator, mod, head, feature, label, label_encoder):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._semclass = list(self.data[feature])
        self._labels = list(self.data[label])
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        self._label_encoder = label_encoder
        self._labels = self._label_encoder.transform(self.labels)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        semclass_embeddings = self.general_extractor.get_array_embeddings(self.semclass)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "label": self.labels[i], "semclass": self.semclass[i],
             "semclass_rep": semclass_embeddings[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def semclass(self):
        return self._semclass

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels(self):
        return self._labels

    @property
    def status(self):
        return self._status

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def samples(self):
        return self._samples


class ClassificationDataset(Dataset):

    def __init__(self, data_path, embedding_path, separator, mod, head, label, label_encoder):
        """
        This datasets can be used to train a composition model on attribute selection
        :param data_path: the path to the dataset, should have a header
        :param embedding_path: the path to the pretrained static word embeddings to lookup the modifier and
        head words
        :param separator: the separator within the dataset
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param label: the name of the column holding the labels (e.g. attribute)
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels = list(self.data[label])
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"

        self._general_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        self._label_encoder = label_encoder
        self._labels = self._label_encoder.transform(self.labels)
        self._samples = self.populate_samples()

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and attributes and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and attribute embeddings (modifier_rep,
        head_rep, attribute_rep)
        """
        mod_embeddings = self.general_extractor.get_array_embeddings(self.modifier_words)
        head_embeddings = self.general_extractor.get_array_embeddings(self.head_words)
        return [
            {"modifier_rep": mod_embeddings[i], "modifier": self.modifier_words[i], "head_rep": head_embeddings[i],
             "head": self.head_words[i], "label": self.labels[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def labels(self):
        return self._labels

    @property
    def status(self):
        return self._status

    @property
    def general_extractor(self):
        return self._general_extractor

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def samples(self):
        return self._samples


class ContextualizedPhraseDataset(Dataset):

    def __init__(self, data_path, bert_model, max_len, lower_case, batch_size, low_layer, top_layer, separator, mod,
                 head,
                 label, label_encoder, context=None, load_bert_embeddings=False, bert_embedding_path=None):
        """
        This Dataset can be used to train a composition model with contextualized embeddings to create attribute-like
        representations
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param label: [String] the label of the column the class label is stored in
        :param mod: [String] the label of the column the modifier is stored in
        :param head: [String] the label of the column the head is stored in
        :param label_definition_path: [String] path to the file that holds the definitions for the labels
        :param context: [String] if given, the dataset should contain a column with context sentences based on which
        the modifier and head words are contextualized
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._path = data_path
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels = list(self.data[label])
        self._label_encoder = label_encoder
        self._labels = self._label_encoder.transform(self.labels)
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"
        self._phrases = [self.modifier_words[i] + " " + self.head_words[i] for i in range(len(self.data))]
        if context:
            self._context_sentences = list(self.data[context])
        else:
            self._context_sentences = [self.modifier_words[i] + " " + self.head_words[i] for i in
                                       range(len(self.data))]

        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size, low_layer=low_layer, top_layer=top_layer)
        self._load_bert_embeddings = load_bert_embeddings
        self._bert_embedding_path = bert_embedding_path

        self._samples = self.populate_samples()

    def lookup_embedding(self, simple_phrases, target_words, low_layer, top_layer):
        return self.feature_extractor.get_single_word_representations(target_words=target_words,
                                                                      sentences=simple_phrases, low_layer=low_layer,
                                                                      top_layer=top_layer)

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and labels and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and phrase embeddings (w1, w2, l)
        """
        fname_modifier = self._path + "_bert_modifier.npy"
        fname_head = self._path + "_bert_head.npy"
        if self.load_bert_embeddings:
            word1_embeddings = np.load(fname_modifier, allow_pickle=True)
            word2_embeddings = np.load(fname_head, allow_pickle=True)
            print("here")
        else:
            print("jetzt")
            word1_embeddings = self.lookup_embedding(target_words=self.modifier_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            word2_embeddings = self.lookup_embedding(target_words=self.head_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            # phrase_embeddings = self.lookup_embedding(target_words=self.phrases,
            #                                          simple_phrases=self.context_sentences,
            #                                          low_layer=10, top_layer=11)
            # phrase_embeddings = []
            # for m, h in zip(word1_embeddings, word2_embeddings):
            #    phrase_embeddings.append(np.mean((np.array(m), np.array(h)), axis=0))

            # phrase_embeddings = self.lookup_embedding(target_words=self.phrases,
            # simple_phrases=self.context_sentences)
            #word1_embeddings = F.normalize(word1_embeddings, p=2, dim=1)
            #word2_embeddings = F.normalize(word2_embeddings, p=2, dim=1)
            print("shape")
            print(np.array(word1_embeddings).shape)

            # np.save(arr=np.array(word1_embeddings), allow_pickle=True, file=fname_modifier)
            # np.save(arr=np.array(word2_embeddings), allow_pickle=True, file=fname_head)

            # label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
            # phrase_embeddings = F.normalize(torch.from_numpy(np.array(phrase_embeddings)), p=2, dim=1)
        return [
            {"modifier_rep": word1_embeddings[i], "modifier": self.modifier_words[i], "head_rep": word2_embeddings[i],
             "head": self.head_words[i], "label": self.labels[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def phrases(self):
        return self._phrases

    @property
    def context_sentences(self):
        return self._context_sentences

    @property
    def labels(self):
        return self._labels

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def samples(self):
        return self._samples

    @property
    def load_bert_embeddings(self):
        return self._load_bert_embeddings

    @property
    def bert_embedding_path(self):
        return self._bert_embedding_path

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def status(self):
        return self._status

    @property
    def path(self):
        return self._path


class ContextualizedSemPhraseDataset(Dataset):

    def __init__(self, data_path, bert_model, max_len, lower_case, batch_size, low_layer, top_layer, separator, mod,
                 head, semclass,
                 label, label_encoder, context=None, load_bert_embeddings=False, bert_embedding_path=None):
        """
        This Dataset can be used to train a composition model with contextualized embeddings to create attribute-like
        representations
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param label: [String] the label of the column the class label is stored in
        :param mod: [String] the label of the column the modifier is stored in
        :param head: [String] the label of the column the head is stored in
        :param label_definition_path: [String] path to the file that holds the definitions for the labels
        :param context: [String] if given, the dataset should contain a column with context sentences based on which
        the modifier and head words are contextualized
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._path = data_path
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._semclass = list(self.data[semclass])
        self._labels = list(self.data[label])
        self._label_encoder = label_encoder
        self._labels = self._label_encoder.transform(self.labels)
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"
        self._phrases = [self.modifier_words[i] + " " + self.head_words[i] for i in range(len(self.data))]
        if context:
            self._context_sentences = list(self.data[context])
        else:
            self._context_sentences = [self.modifier_words[i] + " " + self.head_words[i] + " " + self.semclass[i] for i
                                       in
                                       range(len(self.data))]

        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size, low_layer=low_layer, top_layer=top_layer)
        self._load_bert_embeddings = load_bert_embeddings
        self._bert_embedding_path = bert_embedding_path

        self._samples = self.populate_samples()

    def lookup_embedding(self, simple_phrases, target_words, low_layer, top_layer):
        return self.feature_extractor.get_single_word_representations(target_words=target_words,
                                                                      sentences=simple_phrases, low_layer=low_layer,
                                                                      top_layer=top_layer)

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and labels and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and phrase embeddings (w1, w2, l)
        """
        fname_modifier = self._path + "semclass_bert_modifier.npy"
        fname_head = self._path + "semclas_bert_head.npy"
        fname_sem = self._path + "semclass_bert_semclass.npy"
        if self.load_bert_embeddings:
            word1_embeddings = np.load(fname_modifier, allow_pickle=True)
            word2_embeddings = np.load(fname_head, allow_pickle=True)
            word3_embeddings = np.load(fname_sem, allow_pickle=True)
            print("here")
        else:
            print("jetzt")
            word1_embeddings = self.lookup_embedding(target_words=self.modifier_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            word2_embeddings = self.lookup_embedding(target_words=self.head_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            word3_embeddings = self.lookup_embedding(target_words=self.semclass,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            # phrase_embeddings = self.lookup_embedding(target_words=self.phrases,
            #                                          simple_phrases=self.context_sentences,
            #                                          low_layer=10, top_layer=11)
            # phrase_embeddings = []
            # for m, h in zip(word1_embeddings, word2_embeddings):
            #    phrase_embeddings.append(np.mean((np.array(m), np.array(h)), axis=0))

            # phrase_embeddings = self.lookup_embedding(target_words=self.phrases,
            # simple_phrases=self.context_sentences)
            word1_embeddings = F.normalize(word1_embeddings, p=2, dim=1)
            word2_embeddings = F.normalize(word2_embeddings, p=2, dim=1)
            word3_embeddings = F.normalize(word3_embeddings, p=2, dim=1)
            print(np.array(word1_embeddings).shape)

            np.save(arr=np.array(word1_embeddings), allow_pickle=True, file=fname_modifier)
            np.save(arr=np.array(word2_embeddings), allow_pickle=True, file=fname_head)
            np.save(arr=np.array(word3_embeddings), allow_pickle=True, file=fname_sem)

            # label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
            # phrase_embeddings = F.normalize(torch.from_numpy(np.array(phrase_embeddings)), p=2, dim=1)
        return [
            {"modifier_rep": word1_embeddings[i], "modifier": self.modifier_words[i], "head_rep": word2_embeddings[i],
             "head": self.head_words[i], "label": self.labels[i], "semclass": self.semclass[i],
             "semclass_rep": word3_embeddings[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def semclass(self):
        return self._semclass

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def phrases(self):
        return self._phrases

    @property
    def context_sentences(self):
        return self._context_sentences

    @property
    def labels(self):
        return self._labels

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def samples(self):
        return self._samples

    @property
    def load_bert_embeddings(self):
        return self._load_bert_embeddings

    @property
    def bert_embedding_path(self):
        return self._bert_embedding_path

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def status(self):
        return self._status

    @property
    def path(self):
        return self._path


class ContextualizedRankingDataset(Dataset):

    def __init__(self, data_path, bert_model, max_len, lower_case, batch_size, low_layer, top_layer, separator, mod,
                 head, label_definition_path,
                 label, context=None, load_bert_embeddings=False, load_label_embeddings=False,
                 bert_embedding_path=None):
        """
        This Dataset can be used to train a composition model with contextualized embeddings to create attribute-like
        representations
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param label: [String] the label of the column the class label is stored in
        :param mod: [String] the label of the column the modifier is stored in
        :param head: [String] the label of the column the head is stored in
        :param label_definition_path: [String] path to the file that holds the definitions for the labels
        :param context: [String] if given, the dataset should contain a column with context sentences based on which
        the modifier and head words are contextualized
        """
        self._data = pd.read_csv(data_path, delimiter=separator, index_col=False)
        self._path = data_path
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._labels = list(self.data[label])
        self._definitions = pd.read_csv(label_definition_path, delimiter="\t", index_col=False)
        self._label2definition = dict(zip(list(self._definitions["label"]), list(self._definitions["definition"])))
        self._label_definitions = [l + ", "  + self._label2definition[l] for l in self.labels]
        if "status" in self.data.columns:
            self._status = list(self.data["status"])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.labels), "invalid input data, different lenghts"
        self._phrases = [self.modifier_words[i] + " " + self.head_words[i] for i in range(len(self.data))]
        if context:
            self._context_sentences = list(self.data[context])
        else:
            self._context_sentences = [self.modifier_words[i] + " " + self.head_words[i] for i in
                                       range(len(self.data))]

        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size, low_layer=low_layer, top_layer=top_layer)
        self._load_bert_embeddings = load_bert_embeddings
        self._load_label_embeddings = load_label_embeddings
        self._bert_embedding_path = bert_embedding_path

        self._samples = self.populate_samples()

    def lookup_embedding(self, simple_phrases, target_words, low_layer, top_layer):
        return self.feature_extractor.get_single_word_representations(target_words=target_words,
                                                                      sentences=simple_phrases, low_layer=low_layer,
                                                                      top_layer=top_layer)

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and labels and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and phrase embeddings (w1, w2, l)
        """
        fname_modifier = self._path + "_bert_modifier.npy"
        fname_head = self._path + "_bert_head.npy"
        fname_label = self._path + "_bert_definitions.npy"
        if self.load_bert_embeddings:
            word1_embeddings = np.load(fname_modifier, allow_pickle=True)
            word2_embeddings = np.load(fname_head, allow_pickle=True)
            print("loaded modifier and head embeddings")
        else:
            print("encoding head and modifier")
            word1_embeddings = self.lookup_embedding(target_words=self.modifier_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            word2_embeddings = self.lookup_embedding(target_words=self.head_words,
                                                     simple_phrases=self.context_sentences,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            word1_embeddings = F.normalize(word1_embeddings, p=2, dim=1)
            word2_embeddings = F.normalize(word2_embeddings, p=2, dim=1)
            np.save(arr=np.array(word1_embeddings), allow_pickle=True, file=fname_modifier)
            np.save(arr=np.array(word2_embeddings), allow_pickle=True, file=fname_head)
        if self.load_label_embeddings:
            label_embeddings = np.load(fname_label, allow_pickle=True)

        else:
            label_embeddings = self.lookup_embedding(target_words=self.labels, simple_phrases=self._label_definitions,
                                                     low_layer=self.feature_extractor.low_layer,
                                                     top_layer=self.feature_extractor.top_layer)
            label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
            np.save(arr=np.array(label_embeddings), allow_pickle=True, file=fname_label)

        return [
            {"modifier_rep": word1_embeddings[i], "modifier": self.modifier_words[i], "head_rep": word2_embeddings[i],
             "head": self.head_words[i], "label": self.labels[i], "label_rep": label_embeddings[i]}
            for i in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def phrases(self):
        return self._phrases

    @property
    def context_sentences(self):
        return self._context_sentences

    @property
    def labels(self):
        return self._labels

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def samples(self):
        return self._samples

    @property
    def load_bert_embeddings(self):
        return self._load_bert_embeddings

    @property
    def bert_embedding_path(self):
        return self._bert_embedding_path

    @property
    def load_label_embeddings(self):
        return self._load_label_embeddings

    @property
    def label2definition(self):
        return self._label2definition

    @property
    def label_definitions(self):
        return self._label_definitions

    @property
    def status(self):
        return self._status

    @property
    def path(self):
        return self._path
