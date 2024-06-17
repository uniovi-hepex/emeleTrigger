plots = {
    #---------------------------#####################---------------------------#
    ############################---------------------############################
    ############################         PLOT        ############################
    ############################        YIELDS       ############################
    #---------------------------#####################---------------------------#
    'muonPtvsEta' : {'variable'   : ['muonPt','muonEta'],
                'bins'       : [100,100],  # can be a number or a list of bin edges
                'extra cuts' : 'True', # Applied at per object level only in this plot
                #Plotting thingies
                'xlabel'     : 'Gen $p_T$ (GeV)',
                'ylabel'     : 'Gen $\eta$',
                'range'     : [[0,400],[-2.4,2.4]],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/muonPtvsEta',
                'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                'type'       : '2D',
   },

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
    'stubsN' : {'variable'   : 'stubNo',
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
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # TO DO should add stubLayer !=1 & stubLayer !=3 & stubLayer !=5
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
  #  'stubEtaSigma' : {'variable'   : 'stubEtaSigma',
  #                    'bins'       : 100,
  #                    'extra cuts' : 'True',
  #                    'xlabel'     : 'stubEtaSigma',
  #                    'ylabel'     : 'Yields',
  #                    'xrange'     : [0,1000],
  #                    'ExtraSpam'  :  "",
  #                    'logY'       : False,
  #                    'logX'       : False,
  #                    'grid'       : True,
  #                    'savename'   : '[PDIR]/[DATASET]/Observables/stubEtaSigma',
  #                    'executer'   : 'plotter',
  #                    'type'       : '1D',
  #  },
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
    'stubLayer': {'variable'   : 'stubLayer',
                        'bins'       : 19,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubLayer',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-0.5,18.5],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/stubLayer',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    
    'inputStubNo' : {'variable'   : 'inputStubNo',
                    'bins'       : 10,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '# stubs (input)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [0.5,10.5],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputstubsN',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubPhi' : {'variable'   : 'inputStubPhi',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\phi$ (input stubs)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-1000,1000],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputstubPhi',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubPhiB' : {'variable'   : 'inputStubPhiB',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\phi_B$ input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-500,500],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubPhiB',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
     'inputStubDeltaEta0' : {'variable'   : 'inputStubDeltaEta0',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 0 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta0',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta1' : {'variable'   : 'inputStubDeltaEta1',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 1 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta1',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta2' : {'variable'   : 'inputStubDeltaEta2',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 2 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta2',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta3' : {'variable'   : 'inputStubDeltaEta3',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 3 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta3',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta4' : {'variable'   : 'inputStubDeltaEta4',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 4 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta4',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta5' : {'variable'   : 'inputStubDeltaEta5',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 5 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta5',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta6' : {'variable'   : 'inputStubDeltaEta6',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 6 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta6',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubDeltaEta7' : {'variable'   : 'inputStubDeltaEta7',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \eta$ 7 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-30,30],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaEta7',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
     'inputStubDeltaPhi0' : {'variable'   : 'inputStubDeltaPhi0',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 0 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi0',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
         'inputStubDeltaPhi1' : {'variable'   : 'inputStubDeltaPhi1',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 1 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi1',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
             'inputStubDeltaPhi2' : {'variable'   : 'inputStubDeltaPhi2',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 2 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi2',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
                 'inputStubDeltaPhi3' : {'variable'   : 'inputStubDeltaPhi3',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 3 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi3',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
                     'inputStubDeltaPhi4' : {'variable'   : 'inputStubDeltaPhi4',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 4 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi4',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
                     'inputStubDeltaPhi5' : {'variable'   : 'inputStubDeltaPhi5',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 5 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi5',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
                     'inputStubDeltaPhi6' : {'variable'   : 'inputStubDeltaPhi6',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 6 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi6',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
                     'inputStubDeltaPhi7' : {'variable'   : 'inputStubDeltaPhi7',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\Delta \phi$ 7 input stubs',
                    'ylabel'     : 'Yields',
                    'xrange'     : [-100,100],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDeltaPhi7',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
    'inputStubEta' : {'variable'   : 'inputStubEta',
                    'bins'       : 100,  # can be a number or a list of bin edges
                    'extra cuts' : 'True', # Applied at per object level only in this plot
                    #Plotting thingies
                    'xlabel'     : '$\eta$ (input stubs)',
                    'ylabel'     : 'Yields',
                    'xrange'     : [0,600],
                    'ExtraSpam'  :  "",
                    'logY'       : False,
                    'logX'       : False,
                    'grid'       : True,
                    'savename'   : '[PDIR]/[DATASET]/Observables/inputStubEta',
                    'executer'   : 'plotter', # Safeguard to not run the plot outside of what was designed for it
                    'type'       : '1D',
    },
     'inputStubQuality' : {'variable'   : 'inputStubQuality',
                      'bins'       : 12,
                      'extra cuts' : 'True',
                      'xlabel'     : 'stubQuality input',
                      'ylabel'     : 'Yields',
                      'xrange'     : [0.5,12.5],
                      'ExtraSpam'  :  "",
                      'logY'       : False,
                      'logX'       : False,
                      'grid'       : True,
                      'savename'   : '[PDIR]/[DATASET]/Observables/inputStubQuality',
                      'executer'   : 'plotter',
                      'type'       : '1D',
    },
   'inputStubBx' : {'variable'   : 'inputStubBx',
                        'bins'       : 5,
                        'extra cuts' : 'True',
                        'xlabel'     : 'stubBx input',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-2,2],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/inputStubBx',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'inputStubDetId' : {'variable'   : 'inputStubDetId',
                'bins'       : 1000,
                'extra cuts' : 'True',
                'xlabel'     : 'stubDetId input',
                'ylabel'     : 'Yields',
                'xrange'     : [0,10000],
                'ExtraSpam'  :  "",
                'logY'       : False,
                'logX'       : False,
                'grid'       : True,
                'savename'   : '[PDIR]/[DATASET]/Observables/inputStubDetId',
                'executer'   : 'plotter',
                'type'       : '1D',
    },
    'inputStubType' : {'variable'   : 'inputStubType',
                        'bins'       : 10,
                        'extra cuts' : 'True',
                        'xlabel'     : 'inputStubType',
                        'ylabel'     : 'Yields',
                        'xrange'     : [0.5,10.5],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/inputStubType',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'inputStubTiming' : {'variable'   : 'inputStubTiming',
                        'bins'       : 21,
                        'extra cuts' : 'True',
                        'xlabel'     : 'inputStubTiming',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-10,10],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/inputStubTiming',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
    'inputStubLogicLayer': {'variable'   : 'inputStubLogicLayer',
                        'bins'       : 19,
                        'extra cuts' : 'True',
                        'xlabel'     : 'inputStubLogicLayer',
                        'ylabel'     : 'Yields',
                        'xrange'     : [-0.5,18.5],
                        'ExtraSpam'  :  "",
                        'logY'       : False,
                        'logX'       : False,
                        'grid'       : True,
                        'savename'   : '[PDIR]/[DATASET]/Observables/inputStubLogicLayer',
                        'executer'   : 'plotter',
                        'type'       : '1D',
    },
}
