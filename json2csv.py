from tqdm import tqdm
import ijson
import pandas as pd


def json2csv(in_file, out_file):

    f = open(in_file, encoding="utf8")
    objects = ijson.items(f, 'articles.item')
    articles = []

    for i, obj in enumerate(tqdm(objects)):
        doc = {}
        doc['pmid'] = obj['pmid']
        doc['title'] = obj['title'].strip()
        doc['abstractText'] = obj['abstractText'].strip()
        doc['meshMajor'] = obj['meshMajor']
        doc['meshID'] = obj['meshID']
        doc['journal_MeSH'] = obj['journal']
        doc['neigh_MeSH'] = obj['neighbors']
        doc['year'] = obj['year']
        articles.append(doc)

    df = pd.DataFrame(articles)
    df.to_csv(out_file, index=False)

    return df
