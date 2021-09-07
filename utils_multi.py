import logging
import random
import re
import string
from operator import itemgetter
from typing import Iterator, TypeVar, Sequence, List, Generator

import numpy as np
import torch
from nltk.corpus import stopwords
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset, Sampler, Subset, DistributedSampler
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)


def text_clean(tokens):

    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    clean_tokens = [w for w in stripped if w.isalpha()]  # remove non alphabetic tokens
    text_nostop = [word for word in clean_tokens if word not in stop_words]  # remove stopwords
    filtered_text = [w for w in text_nostop if len(w) > 1]  # remove single character token

    return filtered_text


def _vocab_iterator(train_text, test_text, train_title=None, test_title=None, ngrams=1, is_multichannel=True):

    tokenizer = get_tokenizer('basic_english')

    for i, text in enumerate(train_text):
        if is_multichannel:
            texts = tokenizer(text + train_title[i])
        else:
            texts = tokenizer(text)
        texts = text_clean(texts)
        yield ngrams_iterator(texts, ngrams)

    for i, text in enumerate(test_text):
        if is_multichannel:
            texts = tokenizer(text + test_title[i])
        else:
            texts = tokenizer(text)
        texts = text_clean(texts)
        yield ngrams_iterator(texts, ngrams)


def _text_iterator(text, title=None, labels=None, mesh_mask=None, ngrams=1, yield_label=False, is_multichannel=True):
    tokenizer = get_tokenizer('basic_english')
    for i, text in enumerate(text):
        if is_multichannel:
            texts = tokenizer(text)
            texts = text_clean(texts)
            heading = tokenizer(title[i])
            heading = text_clean(heading)
            mask = mesh_mask[i]
            if yield_label:
                label = labels[i]
                yield label, mask, ngrams_iterator(texts, ngrams), ngrams_iterator(heading, ngrams)
            else:
                yield mask, ngrams_iterator(texts, ngrams), ngrams_iterator(heading, ngrams)
        else:
            texts = tokenizer(text)
            texts = text_clean(texts)
            mask = mesh_mask[i]
            if yield_label:
                label = labels[i]
                yield label, mask, ngrams_iterator(texts, ngrams)
            else:
                yield mask, ngrams_iterator(texts, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk, is_test=False, is_multichannel=True):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        if is_multichannel:
            if is_test:
                for mask, text, title in iterator:
                    if include_unk:
                        ab_tokens = torch.tensor([vocab[token] for token in text])
                        title_tokens = torch.tensor([vocab[token] for token in title])
                    else:
                        ab_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                                  for token in text]))
                        ab_tokens = torch.tensor(ab_token_ids)
                        title_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                                     for token in title]))
                        title_tokens = torch.tensor(title_token_ids)
                    if len(ab_tokens) == 0:
                        logging.info('Row contains no tokens.')
                    data.append((mask, ab_tokens, title_tokens))
                    t.update(1)
                return data
            else:
                for label, mask, text, title in iterator:
                    if include_unk:
                        text = [token for token in text]
                        ab_tokens = torch.tensor([vocab[token] for token in text])
                        title_tokens = torch.tensor([vocab[token] for token in title])
                    else:
                        ab_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in text]))
                        ab_tokens = torch.tensor(ab_token_ids)
                        title_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in title]))
                        title_tokens = torch.tensor(title_token_ids)
                    if len(ab_tokens) == 0:
                        logging.info('Row contains no tokens.')
                    data.append((label, mask, ab_tokens, title_tokens))
                    labels.extend(label)
                    t.update(1)
                return data, list(set(labels))
        else:
            if is_test:
                for mask, text in iterator:
                    if include_unk:
                        tokens = torch.tensor([vocab[token] for token in text])
                    else:
                        token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                               for token in text]))
                        tokens = torch.tensor(token_ids)
                    if len(tokens) == 0:
                        logging.info('Row contains no tokens.')
                    data.append((mask, tokens))
                    t.update(1)
                return data
            else:
                for label, mask, text in iterator:
                    if include_unk:
                        tokens = torch.tensor([vocab[token] for token in text])
                    else:
                        token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                               for token in text]))
                        tokens = torch.tensor(token_ids)
                    if len(tokens) == 0:
                        logging.info('Row contains no tokens.')
                    data.append((label, mask, tokens))
                    labels.extend(label)
                    t.update(1)
                return data, list(set(labels))


class MultiLabelTextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data, labels=None):
        """Initiate text-classification dataset.
         Arguments:
             vocab: Vocabulary object used for dataset.
             data: a list of label/tokens tuple. tokens are a tensor after numericalizing the string tokens.
                   label is a list of list.
                 [([label1], ab_tokens1, title_tokens1), ([label2], ab_tokens2, title_tokens2), ([label3], ab_tokens3, title_tokens3)]
             label: a set of the labels.
                 {label1, label2}
        """
        super(MultiLabelTextClassificationDataset, self).__init__()
        self._vocab = vocab
        self._data = data
        self._labels = labels

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _setup_datasets(train_text, train_labels, test_text, test_labels, train_mask, test_mask, train_title=None, test_title=None, ngrams=1, vocab=None,
                    include_unk=False, is_test=False, is_multichannel=True):
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_text))
        if is_multichannel:
            vocab = build_vocab_from_iterator(_vocab_iterator(train_text, test_text, train_title, test_title, ngrams))
        else:
            vocab = build_vocab_from_iterator(_vocab_iterator(train_text, test_text, ngrams, is_multichannel=is_multichannel))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    if is_multichannel:
        train_data, train_labels = _create_data_from_iterator(
            vocab, _text_iterator(train_text, train_title, labels=train_labels, mesh_mask=train_mask, ngrams=ngrams,
                                  yield_label=True, is_multichannel=is_multichannel), include_unk, is_test=False, is_multichannel=is_multichannel)
        logging.info('Creating testing data')
        if is_test:
            test_data = _create_data_from_iterator(
            vocab, _text_iterator(test_text, test_title, labels=None, mesh_mask=test_mask, ngrams=ngrams, yield_label=False, is_multichannel=is_multichannel), include_unk,
            is_test=is_test, is_multichannel=is_multichannel)
            logging.info('Total number of labels in training set:'.format(len(train_labels)))
            return (MultiLabelTextClassificationDataset(vocab, train_data, train_labels),
                    MultiLabelTextClassificationDataset(vocab, test_data))
        else:
            test_data, test_labels = _create_data_from_iterator(
                vocab, _text_iterator(test_text, test_title, labels=test_labels, mesh_mask=test_mask, ngrams=ngrams, yield_label=True, is_multichannel=is_multichannel), include_unk,
                is_test=False, is_multichannel=is_multichannel)
            logging.info('Total number of labels in training set:'.format(len(train_labels)))
            return (MultiLabelTextClassificationDataset(vocab, train_data, train_labels),
                    MultiLabelTextClassificationDataset(vocab, test_data, test_labels))
    else:
        train_data, train_labels = _create_data_from_iterator(
            vocab, _text_iterator(train_text, labels=train_labels, mesh_mask=train_mask, ngrams=ngrams, yield_label=True, is_multichannel=is_multichannel), include_unk,
            is_test=False, is_multichannel=is_multichannel)
        logging.info('Creating testing data')
        test_data, test_labels = _create_data_from_iterator(
            vocab, _text_iterator(test_text, labels=test_labels, mesh_mask=test_mask, ngrams=ngrams, yield_label=True, is_multichannel=is_multichannel), include_unk,
            is_test=False, is_multichannel=is_multichannel)
        logging.info('Total number of labels in training set:'.format(len(train_labels)))
        return (MultiLabelTextClassificationDataset(vocab, train_data, train_labels),
                MultiLabelTextClassificationDataset(vocab, test_data, test_labels))


def MeSH_indexing(train_text, train_labels, test_text, train_mask, test_mask, test_labels=None, train_title=None, test_title=None, ngrams=1, vocab=None,
                  include_unk=False, is_test=False, is_multichannel=True):
    """

    Defines MeSH_indexing datasets.
    The label set contains all mesh terms in 2019 version (https://meshb.nlm.nih.gov/treeView)


    """
    if is_multichannel:
        if is_test:
            return _setup_datasets(train_text, train_labels, test_text, None, train_mask, test_mask, train_title, test_title, ngrams,
                                   vocab, include_unk, is_test=True, is_multichannel=True)
        else:
            return _setup_datasets(train_text, train_labels, test_text, test_labels, train_mask, test_mask, train_title, test_title, ngrams, vocab,
                                   include_unk, is_multichannel=True)
    else:
        return _setup_datasets(train_text, train_labels, test_text, test_labels, train_mask, test_mask, ngrams=ngrams, vocab=vocab,
                               include_unk=include_unk, is_multichannel=False)


def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if max_len < ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


# torchtext 0.6.0 rewrite torchtext.data.Dataset to inherit torch.data.utils.Dataset
# class TextMultiLabelDataset(data.Dataset):
#     def __init__(self, df, text_field, label_field, txt_col, lbl_cols, **kwargs):
#         # torchtext Field objects
#         fields = [('text', text_field), ('label', label_field)]
#         # for l in lbl_cols:
#         #     fields.append((l, label_field))
#
#         is_test = False if lbl_cols[0] in df.columns else True
#         n_labels = len(lbl_cols)
#
#         examples = []
#         for idx, row in df.iterrows():
#             if not is_test:
#                 lbls = [row[l] for l in lbl_cols]
#             else:
#                 lbls = [0.0] * n_labels
#
#             txt = str(row[txt_col])
#             examples.append(data.Example.fromlist([txt, lbls], fields))
#
#         super(TextMultiLabelDataset, self).__init__(examples, fields, **kwargs)


def text_preprocess(string):
    """
    This method is used to preprocess text

    1. percentage XX% convert to "PERCENTAGE"
    2. Chemical numbers(word contains both number and letters) to "Chem"
    3. All numbers convert to "NUM"
    4. Mathematical symbol （=, <, >, >/=, </= ）
    5. "-" replace with "_"
    6. remove punctuation
    7. covert to lowercase

    """
    string = re.sub("\\d+(\\.\\d+)?%", "percentage", string)
    string = re.sub("((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)", "chemical", string)
    string = re.sub(r'[0-9]+', 'Num', string)
    string = re.sub("=", "Equal", string)
    string = re.sub(">", "Greater", string)
    string = re.sub("<", "Less", string)
    string = re.sub(">/=", "GreaterAndEqual", string)
    string = re.sub("</=", "LessAndEqual", string)
    string = re.sub("-", "_", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("[.,?;*!%^&+():\[\]{}]", " ", string)
    string = string.replace('"', '')
    string = string.replace('/', '')
    string = string.replace('\\', '')
    string = string.replace("'", '')
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def _text_iterator_for_mesh_mask(texts, idfs, labels=None, ngrams=1, yield_label=False):
    tokenizer = get_tokenizer('basic_english')
    table = str.maketrans('', '', string.punctuation)

    for i, text in enumerate(texts):
        tokens = tokenizer(text)
        stripped = [w.translate(table) for w in tokens]  # remove punctuation
        clean_tokens = [w for w in stripped if w.isalpha()]  # remove non alphabetic tokens
        text_nostop = [word for word in clean_tokens if word not in stop_words]  # remove stopwords
        filtered_text = [w for w in text_nostop if len(w) > 1]  # remove single character token
        if yield_label:
            label = labels[i]
            idf = idfs[i]
            yield label, ngrams_iterator(filtered_text, ngrams), idf
        else:
            yield ngrams_iterator(filtered_text, ngrams)


def _create_data_from_iterator_mesh_mask(vocab, iterator, include_unk, is_test=False):
    data = []
    labels = []
    idfs = []
    with tqdm(unit_scale=0, unit='lines') as t:
        if is_test:
            for text, idf in iterator:
                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in text])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in text]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append(tokens, idf)
                idfs.append(idf)
                t.update(1)
            return data, idfs
        else:
            for label, text, idf in iterator:
                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in text])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                           for token in text]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append((label, tokens, idf))
                labels.extend(label)
                idfs.append(idf)
                t.update(1)
            return data, list(set(labels)), idfs


class MeSHMaskDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data, idfs, labels=None):
        """Initiate text-classification dataset.
         Arguments:
             vocab: Vocabulary object used for dataset.
             data: a list of label/tokens tuple. tokens are a tensor after numericalizing the string tokens.
                   label is a list of list.
                 [([label], tokens1), (label2, tokens2), (label2, tokens3)]
             label: a set of the labels.
                 {label1, label2}
        """
        super(MeSHMaskDataset, self).__init__()
        self._vocab = vocab
        self._data = data
        self._labels = labels
        self._idfs = idfs

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab

    def get_idfs(self):
        return self._idfs


def _setup_mesh_mask(train_text, idfs, train_labels, ngrams=1, vocab=None, include_unk=False):
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_text))
        vocab = build_vocab_from_iterator(_text_iterator_for_mesh_mask(train_text, idfs, labels=train_labels, ngrams=ngrams, yield_label=False))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels, train_idfs = _create_data_from_iterator_mesh_mask(
        vocab, _text_iterator_for_mesh_mask(train_text, idfs, labels=train_labels, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    return MeSHMaskDataset(vocab, train_data, train_idfs, train_labels)


def Preprocess(text, idfs, labels, ngrams=1, vocab=None, include_unk=False):
    return _setup_mesh_mask(text, idfs, labels, ngrams, vocab, include_unk)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=False
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# class Subset(Dataset):
#     r"""
#     Subset of a dataset at specified indices.
#     Args:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#     dataset: Dataset
#     indices: Sequence
#
#     def __init__(self, dataset: Dataset, indices: Sequence) -> None:
#         self.dataset = dataset
#         self.indices = indices
#
#     def __getitem__(self, idx):
#         new_idx = self.indices[idx]
#         # if isinstance(idx, list):
#         #     return self.dataset[[self.indices[i] for i in idx]]
#
#         return self.dataset[new_idx]
#
#     def __len__(self):
#         print('subset length', len(self.indices))
#         return len(self.indices)
#
#
# def random_split(dataset: Dataset, lengths: Sequence) -> Subset:
#     r"""
#     Randomly split a dataset into non-overlapping new datasets of given lengths.
#     Optionally fix the generator for reproducible results, e.g.:
#     >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
#     Args:
#         dataset (Dataset): Dataset to be split
#         lengths (sequence): lengths of splits to be produced
#         generator (Generator): Generator used for the random permutation.
#     """
#     # Cannot verify that dataset is Sized
#     if sum(lengths) != len(dataset):
#         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
#
#     indices = randperm(sum(lengths), generator=default_generator).tolist()
#     return ([Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)],
#             [indices[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)])
#
#
# class MultilabelBalancedRandomSampler(Sampler):
#     """
#     MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
#     number of classes n_classes, samples from the data with equal probability per class
#     effectively oversampling minority classes and undersampling majority classes at the
#     same time. Note that using this sampler does not guarantee that the distribution of
#     classes in the output samples will be uniform, since the dataset is multilabel and
#     sampling is based on a single class. This does however guarantee that all classes
#     will have at least batch_size / n_classes samples as batch_size approaches infinity
#     """
#
#     def __init__(self, labels, num_labels, num_examples, class_indices, mlb, train_indices=None, class_choice="least_sampled"):
#         """
#         Parameters:
#         -----------
#             labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
#             indices: an arbitrary-length 1-dimensional numpy array representing a list
#             of indices to sample only from
#             class_choice: a string indicating how class will be selected for every
#             sample:
#                 "least_sampled": class with the least number of sampled labels so far
#                 "random": class is chosen uniformly at random
#                 "cycle": the sampler cycles through the classes sequentially
#         """
#         self.labels = labels
#         self.num_labels = num_labels
#         self.num_examples = num_examples
#         self.indices = train_indices
#         if self.indices is None:
#             self.indices = range(num_examples)
#
#         # List of lists of example indices per class
#         self.class_indices = class_indices
#         self.mlb = mlb
#
#         self.counts = [0] * self.num_labels
#
#         assert class_choice in ["least_sampled", "random", "cycle"]
#         self.class_choice = class_choice
#         self.current_class = 0
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count >= len(self.indices):
#             raise StopIteration
#         self.count += 1
#         print('smaple', self.sample())
#         return self.sample()
#
#     def sample(self):
#         class_ = self.get_class()
#         class_indices = self.class_indices[class_]
#         chosen_index = np.random.choice(class_indices)
#         if self.class_choice == "least_sampled":
#             label_transform = self.mlb.fit_transform([self.labels[chosen_index]])
#             for class_, indicator in enumerate(label_transform[0]):
#                 if indicator == 1:
#                     self.counts[class_] += 1
#         return chosen_index
#
#     def get_class(self):
#         if self.class_choice == "random":
#             class_ = random.randint(0, self.num_labels - 1)
#         elif self.class_choice == "cycle":
#             class_ = self.current_class
#             self.current_class = (self.current_class + 1) % self.num_labels
#         elif self.class_choice == "least_sampled":
#             min_count = self.counts[0]
#             min_classes = [0]
#             for class_ in range(1, self.num_labels):
#                 if self.counts[class_] < min_count:
#                     min_count = self.counts[class_]
#                     min_classes = [class_]
#                 if self.counts[class_] == min_count:
#                     min_classes.append(class_)
#             class_ = np.random.choice(min_classes)
#         return class_
#
#     def __len__(self):
#         return len(self.indices)
