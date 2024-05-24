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
                'type'       : '1D',
   },
   'muonEta' : {'variable'   : 'muonEta',
                'bins'       : 100,  # can be a number or a list of bin edges
                'extra cuts' : 'True', # Applied at per object level only in this plot
                #Plotting thingies
                'xlabel'     : 'Gen $\eta$',
                'ylabel'     : 'Yields',
                'xrange'     : [-2.4,2.4],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/muonEta',
                'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                'type'       : '1D',
   },
   'muonPhi' : {'variable'   : 'muonPhi',
                'bins'       : 100,  # can be a number or a list of bin edges
                'extra cuts' : 'True', # Applied at per object level only in this plot
                #Plotting thingies
                'xlabel'     : 'Gen $\phi$',
                'ylabel'     : 'Yields',
                'xrange'     : [-3.14,3.14],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/muonPhi',
                'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                'type'       : '1D',
   },
   'muonPropEta' : {'variable'   : 'muonPropEta',
                'bins'       : 100,  # can be a number or a list of bin edges
                'extra cuts' : 'True', # Applied at per object level only in this plot
                #Plotting thingies
                'xlabel'     : 'Prop $\eta$',
                'ylabel'     : 'Yields',
                'xrange'     : [-2.4,2.4],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/muonPropEta',
                'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                'type'       : '1D',
   },
    'muonPropPhi' : {'variable'   : 'muonPropPhi',
                 'bins'       : 100,  # can be a number or a list of bin edges
                 'extra cuts' : 'True', # Applied at per object level only in this plot
                 #Plotting thingies
                 'xlabel'     : 'Prop $\phi$',
                 'ylabel'     : 'Yields',
                 'xrange'     : [-3.14,3.14],
                 'ExtraSpam'  :  "",
                 'logY'       : False,
                 'logX'       : False,
                 'grid'       : True,
                 'savename'   : '[PDIR]/[DATASET]/Observables/muonPropPhi',
                 'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                 'type'       : '1D',
    },
    'muonCharge' : {'variable'   : 'muonCharge',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'Charge',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-2,2],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/muonCharge',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'muonDxy' : {'variable'   : 'muonDxy',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'dxy',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/muonDxy',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'muonRho' : {'variable'   : 'muonRho',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'rho',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/muonRho',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'omtfPt' : {'variable'   : 'omtfPt',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'OMTF $p_T$ (GeV)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [0,400],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/omtfPt',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'omtfEta' : {'variable'   : 'omtfEta',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'OMTF $\eta$',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-2.4,2.4],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/omtfEta',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'omtfPhi' : {'variable'   : 'omtfPhi',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'OMTF $\phi$',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-3.14,3.14],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/omtfPhi',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
        },
    'omtfCharge' : {'variable'   : 'omtfCharge',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : 'OMTF Charge',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-2,2],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/omtfCharge',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'stubsN' : {'variable'   : 'nStubs',
                    'bins'       : 10,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '# stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [0.5,10.5],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/stubsN',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'stubPhi' : {'variable'   : 'stubPhi',
                    'bins'       : 1000,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\phi$ (stubs)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-1000,1000],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/stubPhi',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'stubPhiB' : {'variable'   : 'stubPhiB',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\phi_B$ stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-500,500],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/stubPhiB',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'stubEta' : {'variable'   : 'stubEta',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\eta$ (stubs)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [0,600],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/stubEta',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'stubEtaSigma' : {'variable'   : 'stubEtaSigma',
                      'bins'       : 100,
                      'extra cuts' : 'True',
                      'xlabel'     : 'stubEtaSigma',
                      'ylabel'     : 'Yields',
                      'xrange'     : [0,1000],
                      'ExtraSpam'  :  "",
                      'logY'       : False,
                      'logX'       : False,
                      'grid'       : True,
                      'savename'   : '[PDIR]/[DATASET]/Observables/stubEtaSigma',
                      'executer'   : 'plotter',
                      'type'       : '1D',
    },
    'stubQuality' : {'variable'   : 'stubQuality',
                      'bins'       : 12,
                      'extra cuts' : 'True',
                      'xlabel'     : 'stubQuality',
                      'ylabel'     : 'Yields',
                      'xrange'     : [0.5,12.5],
                      'ExtraSpam'  :  "",
                      'logY'       : False,
                      'logX'       : False,
                      'grid'       : True,
                      'savename'   : '[PDIR]/[DATASET]/Observables/stubQuality',
                      'executer'   : 'plotter',
                      'type'       : '1D',
    },
   'stubBx' : {'variable'   : 'stubBx',
                        'bins'       : 5,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubBx',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-2,2],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/stubBx',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'stubDetId' : {'variable'   : 'stubDetId',
                'bins'       : 1000,
                'extra cuts' : 'True',
                'xlabel'     : 'stubDetId',
                'ylabel'     : 'Yields',
                'xrange'     : [0,10000],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/stubDetId',
                'executer'   : 'plotter',
                'type'       : '1D',
    },
    'stubType' : {'variable'   : 'stubType',
                        'bins'       : 10,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubType',
                        'ylabel'     : 'Yields',
                        'xrange'     : [0.5,10.5],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/stubType',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'stubTiming' : {'variable'   : 'stubTiming',
                        'bins'       : 21,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubTiming',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-10,10],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/stubTiming',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'stubLogicLayer': {'variable'   : 'stubLogicLayer',
                        'bins'       : 19,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubLogicLayer',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-0.5,18.5],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/stubLogicLayer',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
}