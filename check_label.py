from sklearn.preprocessing import MultiLabelBinarizer


def get_index_to_meshid(indx = -1, class_name = '', meshIDs=[]):
    if meshIDs == []:
        mapping_id = {}
        with open("/KenMeSH-master/meSH_pair.txt", 'r') as f:
            for line in f:
                (key, value) = line.split('=')
                mapping_id[key] = value.strip()
        meshIDs = list(mapping_id.values())

    # print("meshIDs: ", meshIDs)

    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)

    if indx != -1:
        return mlb.classes_[indx]
    else:
        return list(mlb.classes_).index(class_name)
     
# print(get_index_to_meshid(indx=13536))    