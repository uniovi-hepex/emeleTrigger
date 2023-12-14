import ROOT
import os
from setTDRStyle import setTDRStyle

ROOT.gROOT.SetBatch(True)

def draw_single_vars(inputfile,outputfolder,treename,plots,cutname=""):
    # Open ROOT file
    assert os.path.isfile(inputfile), print('File is does not exist')
    
    file = ROOT.TFile(inputfile)
    tree = file.Get(treename)

    # Draw all branches
    if plots == "all":
        plot_list = []
        for branch in tree.GetListOfBranches():
            if 'hits' in branch.GetName(): continue
            if 'killed' in branch.GetName(): continue
            plot_list.append(branch.GetName())
    else:
        plot_list = plots.split(',')

    for plot in plot_list:
        c1 = ROOT.TCanvas()
        tree.Draw(plot,cutname)
        c1.SaveAs(outputfolder + plot + ".png")

# Draw correlations
def draw_correlations(inputfile,outputfolder,treename,correlations,cutname=""):
    # Open ROOT file
    assert os.path.isfile(inputfile), print('File is does not exist')
    
    file = ROOT.TFile(inputfile)
    tree = file.Get(treename)

    ROOT.gStyle.SetPalette(1)
    c1 = ROOT.TCanvas()
    tree.Draw("muonEta:muonPhi", cutname, "colz")
    c1.SaveAs(outputfolder + "muonEta_vs_muonPhi.png")

    tree.Draw("muonEta:muonPt", cutname, "colz")
    c1.SaveAs(outputfolder + "muonEta_vs_muonPt.png")

    tree.Draw("muonPhi:muonPt", cutname, "colz")
    c1.SaveAs(outputfolder + "muonPhi_vs_muonPt.png")

    tree.Draw("stubPhi:stubProc", cutname, "colz")
    c1.SaveAs(outputfolder + "stubPhi_vs_stubProc.png")

    tree.Draw("stubPhi:stubType", cutname, "colz")
    c1.SaveAs(outputfolder + "stubPhi_vs_stubType.png")

    tree.Draw("stubProc:stubType", cutname, "colz")
    c1.SaveAs(outputfolder + "stubProc_vs_stubType.png")

    tree.Draw("stubPhi:stubQuality", cutname, "colz")
    c1.SaveAs(outputfolder + "stubPhi_vs_stubQuality.png")

    tree.Draw("stubProc:omtfProcessor", cutname, "colz")
    c1.SaveAs(outputfolder + "stubProc_vs_omtfProcessor.png")

    tree.Draw("stubTiming:stubQuality", cutname, "colz")
    c1.SaveAs(outputfolder + "stubTiming_vs_stubQuality.png")

    tree.Draw("stubTiming:stubType", cutname, "colz")
    c1.SaveAs(outputfolder + "stubTiming_vs_stubType.png")

#### ========= MAIN =======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="draw_variables.py [options]",description="Compute mass resolution",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i","--ifile", dest="inputfile",default="data/omtfAnalysis2.root", help='Input filename')
    parser.add_argument("-o","--ofolder",dest="output", default="output/omtfAnalysis2/", help='folder name to store results')
    parser.add_argument("-d","--debug",dest="debug", default=False, action="store_true", help='debug')
    parser.add_argument("-t","--tree",dest="tree", default="simOmtfDigis/OMTFHitsTree", help='tree name')
    parser.add_argument("-p","--plots",dest="plots", default="all", help='plots to be made')
    parser.add_argument("-c","--correlations",dest="correlations", default="all", help='correlations to be made')

    args = parser.parse_args()
    
    inputfile = args.inputfile
    output=args.output

    # Output folder
    if not os.path.exists(output):
        os.makedirs(output);
        os.system("git clone https://github.com/musella/php-plots.git "+output)
    
    print ("Running on: %s " %(inputfile))
    print ("Saving result in: %s" %(output))

    draw_single_vars(inputfile,output,args.tree,args.plots,"muonPropEta!=0&&muonPropPhi!=0")
    draw_correlations(inputfile,output,args.tree,args.correlations,"muonPropEta!=0&&muonPropPhi!=0")
    print ("DONE")

