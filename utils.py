import re
from torchtext import data
import spacy
from tqdm import tqdm


class TextMultiLabelDataset(data.Dataset):
    def __init__(self, text, text_field, label_field, lbls=None, **kwargs):
        # torchtext Field objects
        fields = [('text', text_field), ('label', label_field)]
        # for l in lbl_cols:
        # fields.append((l, label_field))

        is_test = True if lbls is None else False
        if is_test:
            pass
        else:
            n_labels = len(lbls)

        examples = []
        for i, lbl in enumerate(tqdm(lbls)):
            if not is_test:
                l = lbl
            else:
                l = [0.0] * n_labels

            txt = text[i]
            examples.append(data.Example.fromlist([txt, l], fields))

        super(TextMultiLabelDataset, self).__init__(examples, fields, **kwargs)


def tokenize(text):
    tokens = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        tokens.append(token.text)
    return tokens


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

