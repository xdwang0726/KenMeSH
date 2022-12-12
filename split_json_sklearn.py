import argparse
import json
import tqdm 
import os

from sklearn.model_selection import train_test_split

def writefile(data, name):
    f = open(name + '.json', 'wb')
    data = data.encode()
    f.write(data)

def split_json_data(data_path):
    dataset = []
    with tqdm.tqdm(total=os.path.getsize(data_path)) as pbar:   
        with open(data_path) as f:
            lines = f.readlines()
            dataset.append(lines)
    
    print("Line: ", lines)
    print("Datasert length: ", len(dataset))
    train, test = train_test_split(dataset, test_size=0.3)
    val, test = train_test_split(test, test_size=0.5)

    train_data = json.dumps(train, sort_keys = True, indent = 4, ensure_ascii = False)
    writefile(train_data, 'train')
    print('Train data saved...')

    test_data = json.dumps(test, sort_keys = True, indent = 4, ensure_ascii = False)
    writefile(test_data, 'test')
    print('Test data saved...')

    val_data = json.dumps(val, sort_keys = True, indent = 4, ensure_ascii = False)
    writefile(val_data, 'val')
    print('Val data saved...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()

    split_json_data(args.data)


if __name__ == "__main__":
    main()