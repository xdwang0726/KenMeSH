import json
import argparse
import random
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()
    data_path = args.data

    f = open("mesh_count.json", encoding="utf8")
    d = json.load(f)
    
    pmid_lst = []
    for doc in d:
        pmid_lst.append(doc["pmid"])

    f.close()

    print("pmid list: ", pmid_lst)
    f = open(data_path, encoding="utf8")
    d = json.load(f)

    objects = d['articles']
    
    data_lst = []
    for obj in objects:
        if obj["pmid"] in pmid_lst:
            # print(True)
            data = {}
            data["pmid"] = obj["pmid"]
            neighs = obj["neighbors"].split(",")
            data["meshMaskCount"] = len(neighs)
            data_lst.append(data)


    with open("neigh_count.json", "w") as outfile:
        json.dump(data_lst, outfile)
        print("file write complete on: ", outfile)


if __name__ == "__main__":
    main()