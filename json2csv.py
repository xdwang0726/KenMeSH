import pandas as pd
from pathlib import Path
import json

# set path to file
p = Path(r'full_pmc_pubmed.json')

# read json
with p.open('r', encoding='utf-8') as f:
    data = json.loads(f.read())

# print(type(data["articles"][0]))
# create dataframe
# df = pd.json_normalize(data)
# df.to_csv('pubmed_pmc_chunked.csv', index=False, encoding='utf-8')


pdObj = pd.DataFrame.from_dict(data["articles"][0], orient='index')
csvData = pdObj.to_csv('full_pmc_pubmed.json.csv', index=False)
# print(pdObj.head())
