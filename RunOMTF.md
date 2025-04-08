# Installation instructions of OMTF 

## CMSSW_14_2_X 
Install latest OMTF branch (typically from Karol)
```
scram project CMSSW CMSSW_14_2_1 -n CMSSW_14_2_1_PhaseII
cd CMSSW_14_2_1_PhaseII/src/
cmsenv
git cms-init
git fetch my-cmssw
git cms-merge-topic -u kbunkow:from-CMSSW_14_2_0_pre2_KB_v1

git cms-addpkg L1Trigger/Phase2L1GMT
git cms-addpkg DataFormats/L1TMuonPhase2
git cms-addpkg L1Trigger/L1TMuon
git cms-addpkg L1Trigger/L1TMuonOverlapPhase1
git cms-addpkg L1Trigger/L1TMuonOverlapPhase2

git clone git@github.com:cms-data/L1Trigger-L1TMuon L1Trigger/L1TMuon/data1
mv L1Trigger/L1TMuon/data1/* L1Trigger/L1TMuon/data 
cd L1Trigger/L1TMuon/data/omtf_config 
scp lxplus.cern.ch:/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_14_x_x/CMSSW_14_2_0_pre2/src/L1Trigger/L1TMuon/data/omtf_config/ExtrapolationFactors_ExtraplMB1nadMB2_R_EtaValueP1Scale_t35.xml ./
scp lxplus.cern.ch:/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_14_x_x/CMSSW_14_2_0_pre2/src/L1Trigger/L1TMuon/data/omtf_config/lutNN_omtfRegression_v430_FP.xml ./
scp lxplus.cern.ch:/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_14_x_x/CMSSW_14_2_0_pre2/src/L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1andMB2RFixedP_ValueP1Scale_DT_2_2_2_t35__classProb17_recalib2.xml ./
scp lxplus.cern.ch:/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_14_x_x/CMSSW_14_2_0_pre2/src/L1Trigger/L1TMuon/data/omtf_config/muonMatcherHists_100files_smoothStdDev_withOvf.root ./
scp lxplus.cern.ch:/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_14_x_x/CMSSW_14_2_0_pre2/src/L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1nadMB2DTQualAndRFixedP_DT_2_2_t30__classProb17_recalib2.xml
```

Running scripts can be found inside ```L1Trigger/L1TMuonOverlapPhase2/test/expert``` but also: 
```
cp -r /afs/cern.ch/user/f/folguera/public/omtf_scripts/* L1Trigger/L1TMuonOverlapPhase2/test/expert/
```
