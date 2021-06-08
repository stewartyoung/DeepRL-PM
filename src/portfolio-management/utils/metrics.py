import numpy as np
from .eps import eps
import pandas as pd

def SharpeRatio(cumulativeResults, freq=252, rfr = 0):
    """Given a set of returns, calculates naive (rfr=0) (6.27) """
    # calculate daily returns
    returns = cumulativeResults.diff()
    returns = returns.iloc[1:]
    sharpe = ((returns.mean()-rfr) / returns.std()) * np.sqrt(freq)
    return sharpe

def MDD(X, isESG=False):
    """Calculate maximum drawdown (6.28)"""
    mdd = 0
    if isESG:
        # not expressed as a percentage since values are around 0 and cause div/0 errors
        peak = X[0]
        for x in X:
            if x > peak: 
                peak = x
            dd = peak - x
            if dd > mdd:
                mdd = dd
        return mdd
        
    else:
        peak = X[0]
        for x in X:
            if x > peak: 
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return mdd