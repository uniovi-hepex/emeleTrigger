import os, re,sys
from math import *
import numpy as np
from array import array
import uproot
import matplotlib.pyplot as plt

from datasets import *
from plots import *

class plotter(object):
    def __init__(self, options = None):
        self.verbose = options.verbose
        self.options = options
        self.filterDatasets()
        self.filterPlots()
        if not os.path.exists(self.options.pdir):
            os.makedirs(self.options.pdir)
            for dataset in self.datasets:
                os.makedirs(self.options.pdir + "/" + dataset)
                os.makedirs(self.options.pdir + "/" + dataset + "/Observables")
            os.system("git clone https://github.com/musella/php-plots.git " + self.options.pdir)

    def filterDatasets(self):
        self.datasets = {}
        if type(self.options.datasetlist) != type([0,1]):
            if self.options.verbose: print("Converting datasetlist to list")
            self.options.datasetlist = (self.options.datasetlist).split(",")
        for key in datasets:
            if key in self.options.datasetlist:
                self.datasets[key] = datasets[key]
        self.ordereddatasets = self.options.datasetlist

    def filterPlots(self):
        self.plots = {}
        for key in plots:
            if re.match(self.options.plotThis, key):
                if "executer" in plots[key].keys() and not(type(self).__name__ == plots[key]["executer"]):
                    print("[WARNING] Plot %s being skipped by executer==%s configuration option"%(key, plots[key]["executer"]))
                    continue
                self.plots[key] = plots[key]

    def loadFiles(self):
        self.df = {}
        for dataset in self.datasets:
            if self.verbose: print("Loaded dataset %s"%dataset)
            branches = uproot.open("%s:%s" %(self.datasets[dataset]["samples"], self.datasets[dataset]["treename"]))  
            if self.verbose:  print(branches.show())
            branches=branches.arrays(library='pd')
            print('Downsampling...')
            self.df[dataset]=branches.sample(frac=self.options.fraction)
            print('Downsampling done')
    
    def loadVariable(self, variable, df, selection, extraCuts):
        cuts = ""
        for var,sel in selection.items(): 
            cuts += "(%s) & "%sel
        cuts += extraCuts
        return df[variable][df.eval(cuts)]

    def doHistograms(self):
        for plot in self.plots:
            if self.verbose: print("Creating histogram %s"%plot)
            for dataset in self.datasets:
                if self.verbose: print("Creating histogram %s for dataset %s"%(plot,dataset))
                xvar = self.loadVariable(self.plots[plot]["variable"],self.df[dataset],self.datasets[dataset]["selection"],self.plots[plot]["extra cuts"])
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
                plt.close()    

    def run(self):
        if self.verbose: print("Loading files....")
        self.loadFiles()
        if self.verbose: print("Initializating histograms....")
        self.doHistograms()
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
    parser.add_argument("-f","--fraction"   , dest="fraction"     , type=float      , default=0.01    , help="Donwnscale the dataset by this factor")

    options = parser.parse_args()

    print("==========================================")
    hM = plotter(options)
    hM.run()
    print("==========================================")
