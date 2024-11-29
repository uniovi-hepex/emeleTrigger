import math


# PHI CONVERSION

NUM_PROCESSORS = 3
NUM_PHI_BINS = 5400

stubPhi = 0

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

globalPhiRad = stubPhiToGlobalPhi(stubPhi, phiZero(0))


# ETA CONVERSION

etaUnit = 0.010875  # =2.61/240

def stubEtaToGlobalEta (stubEta):
    globalEta = stubEta * etaUnit
    return globalEta

# R conversion

