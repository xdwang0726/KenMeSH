import argparse
import json
import ijson
import tqdm 
import os
import math
import random

def writefile(data, name):
    tmp = { "articles": []}
    tmp["articles"] = data
    
    with open(f"{name}.json", "w") as f_out:
        json.dump(tmp, f_out, indent=4)

def split_json_data(data_path):
    f = open(data_path)
  
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    pmc_data_objects = data['articles']
        
    # pmc_data = json.load(lines)

    items_count = len(pmc_data_objects)
    train_count = math.floor(items_count * 0.8)
    test_count = math.ceil((items_count - train_count) * 0.5)
    val_count = items_count - train_count - test_count

    counters = {}
    train_data = []
    test_data = []
    val_data = []
    remaining = items_count
    counter  = 0

    while True:
        print(f'Expected Total Count: {items_count}, Remainig: {remaining}, Train Count remaining: {train_count - len(train_data)},\
            {len(train_data) <= train_count}, \n Test Count remaining: {test_count - len(test_data)}, Val Count: {val_count - len(val_data)}')
        counter += 1

        num = random.randint(0, items_count-1)
        if counter > 10000000:
            break

        if num in counters:
            continue
        

        remaining = items_count - len(counters)
        if train_count - len(train_data):
            print(f"Pushing Train Data index: {num}")
            train_data.append(pmc_data_objects[num])
            counter = 0
            counters[num] = 1
        elif test_count - len(test_data):
            print(f"Pushing Test Data index: {num}")
            test_data.append(pmc_data_objects[num])
            counter = 0
            counters[num] = 1
        elif val_count - len(val_data):
            print(f"Pushing Val Data index: {num}")
            val_data.append(pmc_data_objects[num])
            counter = 0
            counters[num] = 1

        else:
            break



    print(f'Total Train Count: {len(train_data)}, \n Total Test Count: {len(test_data)}, Total Val Count: {len(val_data)}')

    print("----Preparinig to write data----")
    
    writefile(train_data, 'train')
    print('Train data saved...')

    writefile(test_data, 'test')
    print('Test data saved...')

    writefile(val_data, 'val')
    print('Val data saved...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()

    split_json_data(args.data)


if __name__ == "__main__":
    main()