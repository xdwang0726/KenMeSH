import argparse
import json
import ijson
from tqdm import tqdm 
import os
import math
import random

from utils import dense_to_sparse

def hook(obj):
    value = obj.get("features")
    if value:
        pbar = tqdm(value)
        for item in pbar:
            pass
            pbar.set_description("Loading")
    return obj

def writefile(data, name):
    tmp = { "articles": []}
    tmp["articles"] = data
    
    with open(f"{name}.json", "w") as f_out:
        json.dump(tmp, f_out, indent=4)

def split_json_data(data_path):
    data = []
    f = open(data_path, encoding="utf8")
    # objects = ijson.items(f, 'articles.item')
    d = json.load(f, object_hook=hook)
    objects = d['articles']
    lst = []
    i = 0

    while i < 10000:  
        r = random.randint(0, len(objects))
        if r in lst:
            continue
        data.append(objects[r])
        b = objects[r]["pmid"]
        print(f"{i}. {b} appended..." )
        i += 1

    writefile(data, 'tenk_pmc')
    print('data saved...')

def convert_mesh_mask(data_path):
    data = []
    f = open(data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')
    # d = json.load(f, object_hook=hook)
    # objects = d['articles']

    for i, obj in enumerate(tqdm(objects)):
    # print(type(objects))
    # for i in tqdm(range(len(objects))):
        # obj = objects[i]
        temp = obj["meshMask"]
        sp = dense_to_sparse(temp)
        obj['meshMask'] = str(sp)
        data.append(obj)
        print(obj["pmid"] + " converted....")

    writefile(data, 'dataset_pmc_sp')
    print('data saved...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()

    split_json_data(args.data)
    # convert_mesh_mask(args.data)


if __name__ == "__main__":
    main()