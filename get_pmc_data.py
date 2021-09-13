import argparse
import ijson
import json
import os
import pickle
import urllib.request
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm


def from_mesh2id(labels_list, mapping_id):
    mesh_id = []
    for mesh in labels_list:
        index = mapping_id.get(mesh.strip())
        if index is None:
            print(index)
            pass
        else:
            mesh_id.append(index.strip())
    return mesh_id


def get_pmids_from_pmc(filelist):

    """read file list from PMC at ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt"""

    pmids = []
    with open(filelist, 'r') as f:
        for line in f:
            info = line.split('\t')
            if len(info) <=3:
                continue
            else:
                pmid = info[3]
                if pmid.startswith('PMID:'):
                    pmid = pmid[5:]
                pmids.append(pmid)
    pmids = list(set(list(filter(None, pmids))))

    return pmids


def get_all_linked_file(url):

    """
    get linked file link list from PubMed Annual Baseline at https://lhncbc.nlm.nih.gov/ii/information/MBR.html

    OR
    use command lines : wget https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2002.html
                        sed -n 's/.*href="\([^"]*\).*/\1/p' 2002.html > file_name.txt
                        wget -i file_name.txt
                        gunzip *.gz
    """

    fp = urllib.request.urlopen(url)
    parser = 'html.parser'
    soup = BeautifulSoup(fp, parser, from_encoding=fp.info().get_param('charset'))

    links = []
    for link in soup.find_all('a', href=True):
        links.append(link)

    return links


def check_if_document_is_mannually_curated(file):
    tree = ET.parse(file)
    root = tree.getroot()
    pmids = []
    for articles in root.findall('PubmedArticle'):
        medlines = articles.find('MedlineCitation')
        if 'IndexingMethod' in medlines.attrib:
            pmid = medlines.find('PMID').text
            # file_name = Path(file).name.strip('.xml')[6:]
            # pmid = file_name[:2] + str(version) + file_name[3:]
            pmids.append(pmid)
        else:
            continue
    pmids = list(set(pmids))
    return pmids


def get_mannually_indexed_pmc(pmid, pmc):
    """
    remove the articles that are automated and curated from the PMC list
    """
    pmids = pickle.load(open(pmid, 'rb'))

    pmcs = []
    with open(pmc, 'r') as f:
        for ids in f:
            pmcs.append(ids.strip())

    diff_pmc = list(set(pmcs) - set(pmids))
    print('number of instance in dataset: %d' % diff_pmc)

    return diff_pmc


def check_if_has_meshID(file):

    tree = ET.parse(file)
    root = tree.getroot()
    pmids_no_mesh = []
    for articles in root.findall('PubmedArticle'):
        medlines = articles.find('MedlineCitation')
        pmid = medlines.find('PMID').text
        if medlines.find('MeshHeadingList') is None:
            pmids_no_mesh.append(pmid)

    print('number of ids without mesh %d' % len(pmids_no_mesh))

    return pmids_no_mesh


def get_data(pmid_path, mapping_path, allMesh):

    pmids = []
    with open(pmid_path, 'r') as f:
        for ids in f:
            pmids.append(ids.strip())

    mapping_id = {}
    with open(mapping_path) as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value

    f = open(allMesh, encoding="utf8", errors='ignore')

    objects = ijson.items(f, 'articles.item')

    dataset = []
    missed_id = []
    for i, obj in enumerate(tqdm(objects)):
        data_point = {}
        ids = obj['pmid']
        if ids in set(pmids):
            try:
                heading = obj['title'].strip()
                heading = heading.translate(str.maketrans('', '', '[]'))
                abstract = obj['abstractText'].strip()
                abstract = abstract.translate(str.maketrans('', '', '[]'))
                if len(heading) == 0 or heading == 'In process':
                    print('paper ', ids, ' does not have title!')
                    continue
                elif len(abstract) == 0:
                    print('paper ', ids, ' does not have abstract!')
                    continue
                else:
                    label = obj["meshMajor"]
                    journal = obj['journal']
                    data_point['pmid'] = ids
                    data_point['title'] = heading
                    data_point['abstractText'] = abstract
                    data_point['meshMajor'] = label
                    data_point['meshId'] = from_mesh2id(label, mapping_id)
                    data_point['journal'] = journal
                    dataset.append(data_point)
            except AttributeError:
                print(obj["pmid"])
        else:
            missed_id.append(ids)
            continue
    pubmed = {'articles': dataset}
    return pubmed, missed_id


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--pmids')
    parser.add_argument('--save')
    parser.add_argument('--save_no_mesh')
    parser.add_argument('--pmid_path')
    parser.add_argument('--mapping_path')
    parser.add_argument('--allMesh')
    parser.add_argument('--save_dataset')
    parser.add_argument('--save_missed')

    args = parser.parse_args()

    pmids_list = []
    with open(args.pmids, 'r') as f:
        for ids in f:
            pmids_list.append(ids.strip())
    print('mannually annoted articles: %d' % len(pmids_list))

    no_mesh = []
    for root, dirs, files in os.walk(args.path):
        for file in tqdm(files):
            filename, extension = os.path.splitext(file)
            if extension == '.xml':
                pmids = check_if_has_meshID(file)
                no_mesh.append(pmids)
    no_mesh_pmid_list = list(set([ids for pmids in no_mesh for ids in pmids]))

    new_pmids = list(set(pmids_list) - set(no_mesh_pmid_list))
    print('Total number of articles %d' % len(new_pmids))
    # #
    pickle.dump(no_mesh_pmid_list, open(args.save_no_mesh, 'wb'))
    #
    with open(args.save, 'w') as f:
        for ids in new_pmids:
            f.write('%s\n' % ids)

    # pubmed, missed_ids = get_data(args.pmid_path, args.mapping_path, args.allMesh)
    #
    # with open(args.save_dataset, "w") as outfile:
    #     json.dump(pubmed, outfile, indent=4)
    # pickle.dump(missed_ids, open(args.save_missed, 'wb'))


if __name__ == "__main__":
    main()




