import pickle
import os.path

def pickleIt(filename, timestamp, proportion=1.0, repeat=0, esgEpsilon=0.0, **kwargs):
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    for name, dataset in kwargs.items():
        # print("key:", name, "value:", dataset)
        dataset.to_pickle(path="results/"+timestamp+ "_"+filename+"_"+name+"_prop"+str(proportion)+"_esgeps"+str(esgEpsilon)+"_repeat"+str(repeat)+".pickle")