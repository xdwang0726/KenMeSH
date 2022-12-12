import json
import argparse
import random
import tqdm

# def hook(obj):
#     value = obj.get("articles")
#     if value:
#         pbar = tqdm(value)
#         for item in pbar:
#             pass
#             pbar.set_description("Loading")
#     return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()
    data_path = args.data

    f = open(data_path, encoding="utf8")
    d = json.load(f)

    objects = d['articles']
    doc_count = len(objects)
    print(f"The file contains {doc_count} number of documents.")
    index_lst = []
    i = 0
    data_lst = []
    while i < 100:
        data = {}
        r = random.randint(0, len(objects)-1)
        if r in index_lst:
            continue
        mesh = objects[r]["meshMask"][0]
        mesh_count = mesh.count(1)
        mesh_id = objects[r]["meshID"]
        pmid = objects[r]["pmid"]
        data["pmid"] = pmid
        data["meshId"] = mesh_id
        data["meshMaskCount"] = mesh_count
        data_lst.append(data)
        print("pmId: ", pmid)
        print("Mesh Id: ", mesh_id)
        print("Mesh count of 1: ", mesh_count)
        i += 1


    with open("mesh_count.json", "w") as outfile:
        json.dump(data_lst, outfile)
        print("file write complete on: ", outfile)


if __name__ == "__main__":
    main()