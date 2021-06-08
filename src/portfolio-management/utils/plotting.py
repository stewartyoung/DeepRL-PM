import matplotlib.pyplot as plt
import os 

def plotIt(dataframe, filename, timestamp, proportion=1.0, repeat=0, esgEpsilon=0.0):
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    graph = dataframe.plot(alpha=0.5)
    if dataframe.name == "Results":
        graph.set_ylabel("portfolio value")
        graph.figure.savefig("plots/"+timestamp+ "_"+
            filename+"_Results_prop"+str(proportion)+"_esgeps"+str(esgEpsilon)+"_repeat"+str(repeat)+".png")
    elif dataframe.name == "ESG":
        graph.set_ylabel("portfolio ESG value")
        graph.figure.savefig("plots/"+timestamp+ "_"+
            filename+"_ESG_prop"+str(proportion)+"_esgeps"+str(esgEpsilon)+"_repeat"+str(repeat)+".png")
    return graph