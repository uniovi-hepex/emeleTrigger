import os
import uproot

def get_test_data():
    if not os.path.isfile('data/omtfAnalysis2.root'):
        os.system('wget http://www.hep.uniovi.es/vischia/omtfsweetlove/omtfAnalysis2.root -P data/')
        print('File downloaded into data/')
    else:
        print('File already exists in data/')

    branches = uproot.open('data/omtfAnalysis2.root:simOmtfPhase2Digis/OMTFHitsTree').arrays(library='pd')
    return branches
