import ijson
from tqdm import tqdm

""" get text(abstract and title) and MeSH terms to each document """
f = open('allMeSH_2019.json', encoding="utf8")

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

""" mapping mesh terms to meshIDs """
mapping_id = {}
with open('MeSH_name_id_mapping_2019.txt') as f:
    for line in f:
        (key, value) = line.split('=')
        mapping_id[key] = value

mesh_id_list = []
for mesh in labels_list:
    new_mesh = []
    for item in mesh:
        index = mapping_id.get(item.strip())
        new_mesh.append(index.strip())
    mesh_id_list.append(new_mesh)

file = open("train_meshID.txt", "w", encoding='utf-8')
for meshID in mesh_id_list:
    allID = '|'.join(meshID)
    file.write(allID.strip() + "\r")
file.close()

file = open("train_text.txt", "w", encoding='utf-8')
for i, txt in enumerate(text_list):
    document = ids_list[i] + "|" + txt
    file.write(document.strip() + "\r")
file.close()

file = open("train_meshList.txt", "w", encoding='utf-8')
for i, mesh in enumerate(labels_list):
    m = ids_list[i] + '||' + '|'.join(mesh)
    file.write(m.strip() + "\r")
file.close()
