import logging
import re

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)


def text_clean(tokens):

    stripped = [w.translate(table) for w in tokens]  # remove punctuation
    clean_tokens = [w for w in stripped if w.isalpha()]  # remove non alphabetic tokens
    text_nostop = [word for word in clean_tokens if word not in stop_words]  # remove stopwords
    filtered_text = [w for w in text_nostop if len(w) > 1]  # remove single character token

    return filtered_text


def _vocab_iterator(train_text, train_title, test_text, test_title, labels=None, ngrams=1, yield_label=False):
    tokenizer = get_tokenizer('basic_english')
    for i, text in enumerate(train_text):
        texts = tokenizer(text + train_title[i] + test_text[i] + test_title)
        texts = text_clean(texts)
        if yield_label:
            label = labels[i]
            yield label, ngrams_iterator(texts, ngrams)
        else:
            yield ngrams_iterator(texts, ngrams)


def _text_iterator(text, title, labels=None, ngrams=1, yield_label=False):
    tokenizer = get_tokenizer('basic_english')
    for i, text in enumerate(text):
        texts = tokenizer(text)
        texts = text_clean(texts)
        heading = tokenizer(title[i])
        heading = text_clean(heading)
        if yield_label:
            label = labels[i]
            yield label, ngrams_iterator(texts, ngrams), ngrams_iterator(heading, ngrams)
        else:
            yield ngrams_iterator(texts, ngrams), ngrams_iterator(heading, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk, is_test=False):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        if is_test:
            for text, title in iterator:
                print(text, title, type(text), type(title))
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
                data.append((ab_tokens, title_tokens))
                t.update(1)
            return data
        else:
            for label, text, title in iterator:
                if include_unk:
                    text = [token for token in text]
                    ab_tokens = torch.tensor([vocab[token] for token in text])
                    title_tokens = torch.tensor([vocab[token] for token in title])
                else:
                    # ab_token_ids = [vocab[token] for token in text if token in vocab.itos]
                    ab_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in text]))
                    ab_tokens = torch.tensor(ab_token_ids)
                    title_token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in title]))
                    # title_token_ids = [vocab[token] for token in title if token in vocab.itos]
                    title_tokens = torch.tensor(title_token_ids)
                if len(ab_tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append((label, ab_tokens, title_tokens))
                labels.extend(label)
                t.update(1)
            return data, set(labels)


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


def _setup_datasets(train_text, train_title, train_labels, test_text, test_title, test_labels, ngrams=1, vocab=None,
                    include_unk=False):
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_text))
        vocab = build_vocab_from_iterator(_vocab_iterator(train_text, train_title, test_text, test_title, train_labels, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _text_iterator(train_text, train_title, labels=train_labels, ngrams=ngrams, yield_label=True),
        include_unk,
        is_test=False)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _text_iterator(test_text, test_title, labels=test_labels, ngrams=ngrams, yield_label=True), include_unk,
        is_test=False)
    logging.info('Total number of labels in training set:'.format(len(train_labels)))
    return (MultiLabelTextClassificationDataset(vocab, train_data, train_labels),
            MultiLabelTextClassificationDataset(vocab, test_data, test_labels))


def MeSH_indexing(train_text, train_title, train_labels, test_text, test_title, test_labels, ngrams=1, vocab=None,
                  include_unk=False):
    """

    Defines MeSH_indexing datasets.
    The label set contains all mesh terms in 2019 version (https://meshb.nlm.nih.gov/treeView)


    """

    return _setup_datasets(train_text, train_title, train_labels, test_text, test_title, test_labels, ngrams, vocab,
                           include_unk)


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


# def tokenize(text):
#     tokens = []
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     for token in doc:
#         tokens.append(token.text)
#     return tokens


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
