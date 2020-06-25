import re
from torchtext import data
import spacy


class TextMultiLabelDataset(data.Dataset):
    def __init__(self, df, text_field, label_field, txt_col, lbl_cols, **kwargs):
        # torchtext Field objects
        fields = [('text', text_field), ('label', label_field)]
        # for l in lbl_cols:
        # fields.append((l, label_field))

        is_test = False if lbl_cols[0] in df.columns else True
        n_labels = len(lbl_cols)

        examples = []
        for idx, row in df.iterrows():
            if not is_test:
                lbls = [row[l] for l in lbl_cols]
            else:
                lbls = [0.0] * n_labels

            txt = str(row[txt_col])
            examples.append(data.Example.fromlist([txt, lbls], fields))

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


def get_label_embeddings():

    return

