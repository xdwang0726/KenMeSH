import argparse
import os
import pickle
import urllib.request
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm


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


def check_if_has_meshID(file, pmid_path):
    pmids = []
    with open(pmid_path, 'r') as f:
        for ids in f:
            pmids.append(ids.strip())

    tree = ET.parse(file)
    root = tree.getroot()
    new_pmids = []
    for articles in root.findall('PubmedArticle'):
        medlines = articles.find('MedlineCitation')
        pmid = medlines.find('PMID').text
        if pmid in pmids and medlines.find('MeshHeadingList') is not None:
            new_pmids.append(pmid)

    return new_pmids


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--pmids')
    parser.add_argument('--save')

    args = parser.parse_args()

    pmid_list = []
    for root, dirs, files in os.walk(args.path):
        for file in tqdm(files):
            filename, extension = os.path.splitext(file)
            if extension == '.xml':
                pmids = check_if_has_meshID(file, args.pmids)
                pmid_list.append(pmids)
    pmid_list = list(set([ids for pmids in pmid_list for ids in pmids]))
    print('Total number of articles %d' % len(pmid_list))
    #
    # pickle.dump(pmid_list, open(args.pmids, 'wb'))

    with open(args.save, 'w') as f:
        for ids in pmid_list:
            f.write('%s\n' % ids)


if __name__ == "__main__":
    main()




