plots = {
    #---------------------------#####################---------------------------#
    ############################---------------------############################
    ############################         PLOT        ############################
    ############################        YIELDS       ############################
    #---------------------------#####################---------------------------#
    'muonPt' : {'variable'   : 'muonPt',
                'bins'       : 100,  # can be a number or a list of bin edges
                'extra cuts' : 'True', # Applied at per object level only in this plot
                #Plotting thingies
                'xlabel'     : 'Gen $p_T$ (GeV)',
                'ylabel'     : 'Yields',
                'xrange'     : [0,400],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/muonPt',
                'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
   }
}