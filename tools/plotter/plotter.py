import os, re,sys
from math import *
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd

from datasets import *
from plots import *

import mplhep as hep   
plt.style.use(hep.style.CMS)
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["font.size"] = 15

class plotter(object):
    def __init__(self, options = None):
        self.verbose = options.verbose
        self.options = options
        self.filterDatasets()
        self.filterPlots()
        os.makedirs(self.options.pdir,exist_ok=True)
        import shutil
        index_file = os.path.expanduser("~folguera/public/utils/index.php")

        for dataset in self.datasets:
            print("Creating directory %s"%(self.options.pdir + "/" + dataset))
            dataset_dir = os.path.join(self.options.pdir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            shutil.copy(index_file, os.path.join(dataset_dir, "index.php"))

        shutil.copy(index_file, os.path.join(self.options.pdir, "index.php"))

    def filterDatasets(self):
        self.datasets = {}
        if type(self.options.datasetlist) != type([0,1]):
            if self.options.verbose: print("Converting datasetlist to list")
            self.options.datasetlist = (self.options.datasetlist).split(",")
        for key in datasets:
            if self.options.verbose: print("Checking dataset %s"%key)   
            if key in self.options.datasetlist or self.options.datasetlist[0] == "all":
                self.datasets[key] = datasets[key]
        self.ordereddatasets = self.options.datasetlist

    def filterPlots(self):
        self.plots = {}
        wildcard = self.options.plotThis.strip().lower()
        for key in plots:
            if wildcard in ("*", "all") or re.match(self.options.plotThis, key):
                if "executer" in plots[key].keys() and not(type(self).__name__ == plots[key]["executer"]):
                    print("[WARNING] Plot %s being skipped by executer==%s configuration option"%(key, plots[key]["executer"]))
                    continue
                self.plots[key] = plots[key]

    def loadFiles(self):
        self.df = {}
        for dataset in self.datasets:
            # Check if dataset["samples"] is a list or a string, if it is a list, keep only the first element
#            if isinstance(self.datasets[dataset]["samples"], list):
#                self.datasets[dataset]["samples"] = self.datasets[dataset]["samples"][0]       
            
            # Check if dataset["samples"] is a list, if it is a list, and read all the files in the list
            if isinstance(self.datasets[dataset]["samples"], list):

                '''file_and_tree = "%s:%s" % (self.datasets[dataset]["samples"][0], self.datasets[dataset]["treename"])
                branches = uproot.open(file_and_tree)
                branches = branches.arrays(library='pd')'''

                files_list = self.datasets[dataset]["samples"]
                treename = self.datasets[dataset]["treename"]
                
                # Usamos iterate y acumulamos los arrays en una lista

                branches_list = []
                for file in files_list:
                    if self.verbose: print("   > reading file %s"%file)
                    file_and_tree = "%s:%s" % (file, treename)
                    try:
                        one_branch = uproot.open(file_and_tree)
                        one_branch = one_branch.arrays(library='pd')
                        # Downsampling...
                        one_branch = one_branch.sample(frac=self.options.fraction)
                        branches_list.append(one_branch)
                    except uproot.KeyInFileError as e:
                        print("[WARNING] Key not found in file %s: %s" % (file_and_tree, e))
                        continue

                # Concatenamos los DataFrames obtenidos, si se cargó alguno
                if branches_list:
                    if self.verbose: print("   > concatenating %d branches"%len(branches_list))
                    branches = pd.concat(branches_list)
                else:
                    print("[ERROR] No branches loaded for dataset %s" % dataset)
                    branches = pd.DataFrame()

            else: 
                if self.verbose: print("Reading files from %s"%self.datasets[dataset]["samples"])
                try:
                    file_and_tree = "%s:%s" % (self.datasets[dataset]["samples"], self.datasets[dataset]["treename"])
                    branches = uproot.open(file_and_tree)
                    branches = branches.arrays(library='pd')
                except uproot.KeyInFileError as e:
                    print("[WARNING] Key not found in file %s: %s" % (file_and_tree, e))
                    branches = pd.DataFrame()
            
            if self.verbose: print("Loaded dataset %s"%dataset)

            if self.verbose: print('Downsampling (twice in case of lists)...')
            self.df[dataset]=branches.sample(frac=self.options.fraction)
            if self.verbose: print('Downsampling done')
    
    def addVariables(self):
        print("Adding variables to datasets")
        for dataset in self.datasets:
            if self.verbose:
                print(f"Agregando variables a {dataset}")
            df = self.df[dataset]
            # Verificar que existan las columnas 'muonCharge' y 'muonPt'
            if "muonCharge" in df.columns and "muonPt" in df.columns:
                df["muonQPt"] = df["muonCharge"] * df["muonPt"]
                df["muonQOverPt"] = df["muonCharge"] / df["muonPt"]
            else:
                print(f"[WARNING] El dataset {dataset} no contiene 'muonCharge' y/o 'muonPt'")
            self.df[dataset] = df 

    def loadVariableFromDataset_tonumpy(self, plot, dataset,index):
        variable = self.plots[plot]["variable"][index]
        cuts = ""
        print(variable)
        for sel in self.datasets[dataset]["selection"].values(): 
            cuts += "(%s) & "%sel
        if "()" in cuts or "(True)" in cuts:
           s = self.df[dataset][variable]
        else:
           cuts += self.plots[plot]["extra cuts"]
           selection = self.df[dataset].eval(cuts) 
           s = self.df[dataset][variable][selection]

        s_nparr = s.to_numpy()
        return s_nparr
        
    def loadVariableFromDataset(self, plot, dataset):
        variable = self.plots[plot]["variable"]
        cuts = ""
        for sel in self.datasets[dataset]["selection"].values(): 
            cuts += "(%s) & "%sel
        if "()" in cuts or "(True)" in cuts:
           print(variable)
           s = self.df[dataset][variable]
        else:
           cuts += self.plots[plot]["extra cuts"]
           selection = self.df[dataset].eval(cuts) 
           s = self.df[dataset][variable][selection]
        if isinstance(s.values[0],ak.Array):
            return pd.Series(ak.flatten(s.values))
        else: 
            return s
        '''if isinstance(df[variable],ak.Array):
            if self.verbose: print("Converting to pandas series") 
            return ak.flatten(df[variable])[ak.flatten(selection)]
        else:
            return df[variable][selection]'''
            
    def plot1DHisto(self,series,plot,dataset):
        fig, ax = plt.subplots()
        series.plot(kind='hist', xlabel=plot["xlabel"], ylabel=plot["ylabel"], bins=plot["bins"], range=plot["xrange"], 
                        label=dataset["label"], color=dataset["color"], logy=plot["logY"], logx=plot["logX"],histtype='step', 
                        grid=plot["grid"],fill=False)
        if dataset["label"] != "": ax.legend()
        '''        
try:
            n, bins, patches = plt.hist(series, bins=plot["bins"], range=plot["xrange"], histtype='step', 
                                        density=self.options.normalize, label=dataset["label"], color=dataset["color"])
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            if plot["logY"]:  ax.set_yscale('log')
            if plot["logX"]:  ax.set_xscale('log')
        except:
            print('[WARNING] Cannot plot %s with matplotlib'%plot["xlabel"])
'''
        fig.tight_layout()
        #plt.show(block=False)
        plt.savefig(plot["savename"].replace("[PDIR]",self.options.pdir).replace("[DATASET]",dataset["name"]))
        plt.close()
        
    def plot2DHisto(self,xarry, yarray,plot,dataset):
        fig, ax = plt.subplots()
        bins=plot["bins"]
        range=plot["range"]
        plt.hist2d(xarry,yarray,bins,range)
        plt.colorbar()
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel(plot["ylabel"])
        #plt.show(block=False)
        plt.savefig(plot["savename"].replace("[PDIR]",self.options.pdir).replace("[DATASET]",dataset["name"]))
        plt.close()

    def plotHistograms(self):
        for plot in self.plots:
            for dataset in self.datasets:
                if self.verbose: print("Creating histogram %s for dataset %s"%(plot,dataset))
                if self.plots[plot]['type'] == '1D':    
                    xvar = self.loadVariableFromDataset(plot,dataset)
                    self.plot1DHisto(xvar,self.plots[plot],self.datasets[dataset])
                elif self.plots[plot]['type'] == '2D':
                    xvar = self.loadVariableFromDataset_tonumpy(plot,dataset,0)
                    yvar = self.loadVariableFromDataset_tonumpy(plot,dataset,1)
                    self.plot2DHisto(xvar,yvar, self.plots[plot],self.datasets[dataset])
                    #print('[WARNING] 2D plots not implemented yet')
                    continue
                    
                '''
                fig, ax = plt.subplots()
                n, bins, patches = plt.hist(xvar, bins=self.plots[plot]["bins"], range=self.plots[plot]["xrange"], histtype='step', density=self.options.normalize, label=self.datasets[dataset]["label"], color=self.datasets[dataset]["color"])
                ax.set_xlabel(self.plots[plot]["xlabel"])
                ax.set_ylabel(self.plots[plot]["ylabel"])
                if self.plots[plot]["logY"]:  ax.set_yscale('log')
                if self.plots[plot]["logX"]:  ax.set_xscale('log')
                if self.datasets[dataset]["label"] != "": ax.legend()
                fig.tight_layout()
                plt.show(block=False)
                plt.savefig(self.plots[plot]["savename"].replace("[PDIR]",self.options.pdir).replace("[DATASET]",dataset))
                plt.close()'''

    def run(self):
        if self.verbose: print("Loading files....")
        self.loadFiles()
        self.addVariables()
        if self.verbose: print("Initializating histograms....")
        self.plotHistograms()
        if self.verbose: print("Done!")

#### ========= MAIN =======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="plotter.py [options]",description="plotter for OMTF studies",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d","--datasets" , dest="datasets"   , type=str      , default="datasets" , help="File to read dataset configuration from")
    parser.add_argument("-l","--datasetlist"  , dest="datasetlist", type=str  ,default="MuGun_FlatPt0To1000Dxy3m", help="Datasets to plot, in order to appear in legend")
    parser.add_argument("-p","--plots"    , dest="plots"      , type=str      , default="plots"    , help="File to read plot configuration from")
    parser.add_argument("-q","--plotlist" , dest="plotThis"      , type=str      , default="muon*"  , help="Plots to be activated")
    parser.add_argument("--pdir"    , dest="pdir"      , type=str      , default="./output/"    , help="Where to put the plots into")
    parser.add_argument("-v","--verbose"  , dest="verbose"    , action="store_true", default=True         , help="If activated, print verbose output")
    parser.add_argument("-n","--normalize"  , dest="normalize"    , action="store_true", default=False         , help="Normalize histograms to unity")
    parser.add_argument("-f","--fraction"   , dest="fraction"     , type=float      , default=0.1    , help="Donwnscale the dataset by this factor")

    options = parser.parse_args()

    print("==========================================")
    hM = plotter(options)
    hM.run()
    print("==========================================")
