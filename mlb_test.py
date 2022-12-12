from sklearn.preprocessing import MultiLabelBinarizer

meshIDs = ['D007558','D055333','D013431','D007606','D008920','D018819','D005704','D052120','D010641','D000792']
neighmesh = ['D055333','D010641','D000792','D006351','D008722','D054990','D008654','D005951','D012394','D012508','D007649']

mlb = MultiLabelBinarizer(classes=meshIDs)
mlb.fit(meshIDs)

m = mlb.fit_transform([neighmesh])

n = mlb.fit_transform(m)

print("M :\n", m)
print("N :\n", n)

