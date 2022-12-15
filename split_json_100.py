import argparse
import json
import ijson
import tqdm 
import os
import math
import random

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

    while i < 100:  
        r = random.randint(0, len(objects))
        if r in lst:
            continue
        data.append(objects[r])
        b = objects[r]["pmid"]
        print(f"{i}. {b} appended..." )
        i += 1

    writefile(data, 'hundred_pmc')
    print('data saved...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()

    split_json_data(args.data)


if __name__ == "__main__":
    main()