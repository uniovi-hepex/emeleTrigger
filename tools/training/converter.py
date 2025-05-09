import math
import torch

NUM_PROCESSORS = 3
NUM_PHI_BINS = 5400
HW_ETA_TO_ETA_FACTOR=0.010875
LOGIC_LAYERS_LABEL_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: 'MB1',
            2: 'MB2',
            4: 'MB3',
            6: 'ME1/3',
            7: 'ME2/2',
            8: 'ME3/2',
            9: 'ME1/2',
            10: 'RB1in',
            11: 'RB1out',
            12: 'RB2in',
            13: 'RB2out',
            14: 'RB3',
            15: 'RE1/3',
            16: 'RE2/3',
            17: 'RE3/3'
        }

def foldPhi (phi):
    if (phi > NUM_PHI_BINS / 2):
        return (phi - NUM_PHI_BINS)
    elif (phi < -NUM_PHI_BINS / 2):
        return (phi + NUM_PHI_BINS)
    return phi

def phiZero (processor):
    phiZero = foldPhi(NUM_PHI_BINS / NUM_PROCESSORS * (processor) + NUM_PHI_BINS / 24)  # Adjusted based on phiZero values
    return phiZero

def stubPhiToGlobalPhi (stubPhi, phiZero):
    globalPhi = foldPhi(stubPhi + phiZero)
    phiUnit = 2 * math.pi / NUM_PHI_BINS
    return globalPhi * phiUnit

def globalPhiToStubPhi (globalPhi, phiZero):
    stubPhi = foldPhi(globalPhi / (2 * math.pi) * NUM_PHI_BINS - phiZero)
    return stubPhi

def get_global_phi(phi, processor):
    p1phiLSB = 2 * math.pi / NUM_PHI_BINS
    if isinstance(phi, list):
        return [(processor * 192 + p + 600) % NUM_PHI_BINS * p1phiLSB for p in phi]
    else:
        return (processor * 192 + phi + 600) % NUM_PHI_BINS * p1phiLSB

def get_stub_r(stubTypes, stubEta, stubLayer, stubQuality):
    rs = []
    for stubType, stubEta, stubLayer, stubQuality in zip(stubTypes, stubEta, stubLayer, stubQuality):
        r = None
        if stubType == 3:  # DTs
            if stubLayer == 0:
                r = 431.133
            elif stubLayer == 2:
                r = 512.401
            elif stubLayer == 4:
                r = 617.946

            # Low-quality stubs are shifted by 23.5/2 cm
            if stubQuality == 2 or stubQuality == 0:
                r = r - 23.5 / 2
            elif stubQuality == 3 or stubQuality == 1:
                r = r + 23.5 / 2

        elif stubType == 9:  # CSCs
            if stubLayer == 6:
                z = 690  # ME1/3
            elif stubLayer == 9:
                z = 700  # M1/2
            elif stubLayer == 7:
                z = 830
            elif stubLayer == 8:
                z = 930
            r = z / np.cos(np.tan(2 * np.arctan(np.exp(-stubEta * HW_ETA_TO_ETA_FACTOR))))
        elif stubType == 5:  # RPCs, but they will be shut down because they leak poisonous gas
            r = 999.
            if stubLayer == 10:
                r = 413.675  # RB1in
            elif stubLayer == 11:
                r = 448.675  # RB1out
            elif stubLayer == 12:
                r = 494.975  # RB2in
            elif stubLayer == 13:
                r = 529.975  # RB2out
            elif stubLayer == 14:
                r = 602.150  # RB3
            elif stubLayer == 15:
                z = 720  # RE1/3
            elif stubLayer == 16:
                z = 790  # RE2/3
            elif stubLayer == 17:
                z = 970  # RE3/3
            if r == 999.:
                r = z / np.cos(np.tan(2 * np.arctan(np.exp(-stubEta * HW_ETA_TO_ETA_FACTOR))))

        rs.append(r)

    if len(rs) != len(stubTypes):
        print('Tragic tragedy. R has len', len(rs), ', stubs have len', len(stubTypes))
    return np.array(rs, dtype=object)
    
def getEtaKey(eta):
    if abs(eta) < 0.92:
        return 1
    elif abs(eta) < 1.1:
        return 2
    elif abs(eta) < 1.15:
        return 3
    elif abs(eta) < 1.19:
        return 4
    else:
        return 5
    
def getListOfConnectedLayers(eta):
    etaKey=getEtaKey(eta)    

    LAYER_ORDER_MAP = {
            1: [10,0,11,12,2,13,14,4,6,15],
            2: [10,0,11,12,2,13,6,15,16,7],
            3: [10,0,11,6,15,16,7,8,17],
            4: [10,0,11,16,7,8,17],
            5: [10,0,9,16,7,8,17],
    }
    return LAYER_ORDER_MAP[etaKey]    

def getEdgesFromLogicLayer(logicLayer,withRPC=True):
    LOGIC_LAYERS_CONNECTION_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: [2,4,6,7,8,9],   #MB1: [MB2, MB3, ME1/3, ME2/2]
            2: [4,6,7],         #MB2: [MB3, ME1/3]
            4: [6],             #MB3: [ME1/3]
            6: [7,8],           #ME1/3: [ME2/2]
            7: [8,9],           #ME2/2: [ME3/2]
            8: [9],             #ME3/2: [RE3/3]
            9: [],              #ME1/2: [RE2/3, ME2/2]
    }
    LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC = {
            0:  [2,4,6,7,8,9,10,11,12,13,14,15,16,17], 
            1:  [2,4,6,7,8,9,10,11,12,13,14,15,16,17], 
            2:  [4,6,7,10,11,12,13,14,15,16],       #MB2: [MB3, ME1/3]
            3:  [4,6,7,10,11,12,13,14,15,16],       #MB2: [MB3, ME1/3]
            4:  [6,10,11,12,13,14,15],         #MB3: [ME1/3]
            5:  [6,10,11,12,13,14,15],         #MB3: [ME1/3]
            6:  [7,8,10,11,12,13,14,15,16,17],         #ME1/3: [ME2/2]
            7:  [8,9,10,11,15,16,17],         #ME2/2: [ME3/2]
            8:  [9,10,11,15,16,17],        #ME3/2: [RE3/3]
            9:  [7,10,16,17],         #ME1/2: [RE2/3, ME2/2]
            10: [11,12,13,14,15,16,17],
            11: [12,13,14,15,16,17],
            12: [13,14,15,16],
            13: [14,15,16],
            14: [15],
            15: [16,17],
            16: [17],
            17: []
    }
        
    if (withRPC):
        return (LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC[logicLayer])
    else:
        if (logicLayer>=10): return []
        return (LOGIC_LAYERS_CONNECTION_MAP[logicLayer])

def remove_empty_or_nan_graphs(data):
    # Verificar si el grafo está vacío
    if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
        return None

    # Verificar si hay valores nan en x, edge_attr o y
    if torch.isnan(data.x).any() or (data.edge_attr is not None and torch.isnan(data.edge_attr).any()) or torch.isnan(data.y).any():
        return None

    return data


