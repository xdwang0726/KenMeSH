import ijson
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--MeshID')
    parser.add_argument('--train_meshID')
    parser.add_argument('--train_text')
    parser.add_argument('--train_MeshList')
    args = parser.parse_args()

    """ get text(abstract and title) and MeSH terms to each document """
    f = open(args.allMesh, encoding="utf8")

    objects = ijson.items(f, 'articles.item')

    ids_list = []
    text_list = []
    labels_list = []

    for obj in tqdm(objects):
        try:
            ids = obj["pmid"].strip()
            text = obj["title"].strip() + " " + obj["abstractText"].strip()
            label = obj["meshMajor"]
            ids_list.append(ids)
            text_list.append(text)
            labels_list.append(label)
        except AttributeError:
            print(obj["pmid"].strip())

    print('Finished Loading Data!')
    """ mapping mesh terms to meshIDs """
    mapping_id = {}
    with open(args.MeshID) as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value

    mesh_id_list = []
    for mesh in labels_list:
        new_mesh = []
        for item in mesh:
            index = mapping_id.get(item.strip())
            if index is None:
                print(index)
                pass
            else:
                new_mesh.append(index.strip())
        mesh_id_list.append(new_mesh)

    print("Writing training MeSH ID to file")
    file = open(args.train_meshID, "w", encoding='utf-8')
    for meshID in mesh_id_list:
        allID = '|'.join(meshID)
        file.write(allID.strip() + "\r")
    file.close()

    print('Writing trianing text to file')
    file = open(args.train_text, "w", encoding='utf-8')
    for i, txt in enumerate(text_list):
        document = ids_list[i] + "|" + txt
        file.write(document.strip() + "\r")
    file.close()

    print('Writing training MeSH list to file')
    file = open(args.train_MeshList, "w", encoding='utf-8')
    for i, mesh in enumerate(labels_list):
        m = ids_list[i] + '||' + '|'.join(mesh)
        file.write(m.strip() + "\r")
    file.close()


if __name__ == "__main__":
    main()
