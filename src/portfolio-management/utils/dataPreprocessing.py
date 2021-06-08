"""
Creates a multiIndexed pandas dataframe for stock data with financial and non-financial features

MultiIndex(levels=[['Stock1', ..., 'StockN'], ['close', 'high', 'low', 'open', 'feature']],
 labels=[[0, 0, 0, 0, 0, ..., N, N, N, N, N], [0, 1, 2, 3, 4, ..., 0, 1, 2, 3, 4]], 
 names=['StockName', 'Price'])
"""
import pandas as pd
import numpy as np
import random


def dateTimeAsIndex(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe = dataframe.set_index('Date')
    return dataframe


def getFeatureColumns(dataframe):
    # get the column names that correspond to features we want
    # 0 = Close, 3 = High, 4 = Low, 5 = Open, 10 = Feature
    features = []
    for i in range(len(dataframe.columns)):
        if i % 11 == 0:
            # close
            features.append(dataframe.columns[i])
        elif i % 11 == 3:
            # high
            features.append(dataframe.columns[i])
        elif i % 11 == 4:
            # low
            features.append(dataframe.columns[i])
        elif i % 11 == 5:
            # open
            features.append(dataframe.columns[i])
        elif i % 11 == 10:
            # feature
            features.append(dataframe.columns[i])
    dataframe = dataframe[features]
    return dataframe


def removeErrorCols(dataframe, removeCols=None):
    dataframe = dataframe[dataframe.columns.drop(
        list(dataframe.filter(regex='ERROR')))]
    if removeCols is not None:
        # get columns not in
        dataframe = dataframe[dataframe.columns.drop(removeCols)]
    return dataframe


def getMultiIndex(dataframe, removeStocks):
    # first get stock names
    # dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex=removeStocks)))]
    stockNames = []
    for i in range(len(dataframe.columns)):
        if i % 5 == 0:
            # print(dataframe.columns[i])
            stockNames.append(dataframe.columns[i])

    featureNames = ['close', 'high', 'low', 'open', 'feature']
    numFeatures = len(featureNames)
    numStocks = dataframe.shape[1]//numFeatures
    stockPandasLabels = []
    for i in range(numStocks):
        for j in range(numFeatures):
            stockPandasLabels.append(i)

    featurePandasLabels = []
    for i in range(numStocks):
        j = 0
        while j < numFeatures:
            featurePandasLabels.append(j)
            j = j + 1

    multi = pd.MultiIndex(levels=[stockNames, featureNames],
                          labels=[stockPandasLabels, featurePandasLabels],
                          names=["Stock", "Feature"])
    return multi  # , dataframe


def getFullCloseAndFeatureColumns(dataframe, stockFeatures):
    count = 0
    featureClean = [stockFeatures[0], stockFeatures[10]]
    stockNames = [list(dataframe.columns.values)[i]
                  for i in range(1, dataframe.shape[1], len(stockFeatures))]
    features = [stockNames[i] + featureClean[j]
                for i in range(len(stockNames)) for j in range(len(featureClean))]

    keepOnly = []
    count = 0
    for i in range(0, len(features), 2):
        if (dataframe[features].iloc[:, i].isnull().any() == False and dataframe[features].iloc[:, i+1].isnull().any() == False):
            keepOnly.append(features[i])
            count += 1
    if len(keepOnly) > 0:
        print("{} out of {} have full close and feature data".format(
            count, len(stockNames)))
        print(keepOnly)
        return keepOnly, count
    else:
        raise Exception("0 stocks have non NaN close and feature columns")


def preProcessData(filepath, removeCols=None, removeStocks=None, keepOnly=None, esgScreen=False, proportionAssets=1.0):
    dataframe = pd.read_csv(filepath)
    stockFeatures = ["", " - DIVIDEND YIELD", " - PER", " - PRICE HIGH", " - PRICE LOW", " - OPENING PRICE", " - UNADJ. PRICE OPEN",
                     " - Environment Pillar Score", " - Governance Pillar Score", " - Social Pillar Score", " - ESG Combined Score"]
    # remove columns by column numbers if heading has errors
    if "CAC40" in filepath:
        # remove TECHNIPFMC (PAR)
        dataframe = dataframe.iloc[:, 0:430]
    if "FTSE100" in filepath:
        # remove shell column with error string causing problems calculating np means etc.
        dataframe = dataframe.iloc[:, np.r_[0:78, 89:dataframe.shape[1]]]
    if "NIKKEI225" in filepath:
        # remove NIPPON SUISAN KAISHA, SCREEN HOLDINGS, SKY PERFECT JSAT HDG., HITACHI ZOSEN,
        # MARUHA NICHIRO, NIPPON LIGHT METAL HDG., PACIFIC METALS, TOHO ZINC, UNITIKA
        dataframe = dataframe.iloc[:, np.r_[0:1783, 1794:1926, 1937:1992, 2003:2344,
                                            2355:2377, 2388:2399, 2410:2432, 2454:2465, 2476:dataframe.shape[1]]]
    if "TSX" in filepath:
        # remove FRONTERA ENERGY
        dataframe = dataframe.iloc[:, np.r_[0:276, 287:dataframe.shape[1]]]
    if "SP500" in filepath:
        # remove DUPONT DE NEMOURS, CORTEVA, AMCOR, FOX A, TECHNIPFMC, FOX B
        dataframe = dataframe.iloc[:, np.r_[0:1684, 1695:1893, 1904:2993,
                                            3004:3950, 3961:4599, 4610:5259, 5270:5523, 5534:dataframe.shape[1]]]

    # remove columns by stock name, e.g., removeStocks=["AAPL","AMZN",...]
    if (removeStocks != None and len(removeStocks) > 0):
        screen = []
        for i in range(len(removeStocks)):
            for j in range(len(stockFeatures)):
                screen.append(removeStocks[i]+stockFeatures[j])
        for col in screen:
            del dataframe[col]

    # keep only remaining columns with full Close and  features
    if esgScreen:
        keepOnly, count = getFullCloseAndFeatureColumns(
            dataframe=dataframe, stockFeatures=stockFeatures)

    # keep only specific stocks, e.g., keepStocks=["AAPL","AMZN",...]
    if (keepOnly != None and len(keepOnly) > 0):
        try:
            num_assets_tokeep = int(count*proportionAssets)
        except:
            num_assets_tokeep = int(dataframe.shape[1]*proportionAssets // 11)
        # take a random sample of size "num_assets_tokeep"
        keepOnly = random.sample(keepOnly, num_assets_tokeep)
        print("Random sample of {}% of full index".format(proportionAssets*100))
        print("Assets randomly chosen: ", keepOnly)
        # conduct final screening of original data set using random sample of stock names
        screen = []
        screen.append("Date")
        for i in range(len(keepOnly)):
            for j in range(len(stockFeatures)):
                screen.append(keepOnly[i]+stockFeatures[j])
        dataframe = dataframe[[c for c in dataframe.columns if c in screen]]

    dataframe['Date'] = pd.to_datetime(dataframe['Date'], dayfirst=True)
    dataframe = dateTimeAsIndex(dataframe)
    dataframe = getFeatureColumns(dataframe)
    """
    Creates a multiIndexed pandas dataframe for stock data with financial and non-financial features

    MultiIndex(levels=[['Stock1', ..., 'StockN'], ['close', 'high', 'low', 'open', 'feature']],
    labels=[[0, 0, 0, 0, 0, ..., N, N, N, N, N], [0, 1, 2, 3, 4, ..., 0, 1, 2, 3, 4]], 
    names=['StockName', 'Price'])
    """
    multiIndex = getMultiIndex(dataframe, removeStocks)
    dataframe.columns = multiIndex
    return dataframe
