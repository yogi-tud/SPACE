import gc
import sys
import os
#launcher for EDBT experiments
# syntax 2 0.01 1024 4 3
# Dataset , selectivity, datasize, cluster count, datatype
# dataset (0 uniform, 1 1cluster, 2 multicluster)
# selectivity % of 1 bits in mask [0 ; 1]
# datasize in MIB of input col
# datatypes 1-uint8 2-uint16 3-uint32 4-int 5-float

def run_sel(sel,dataset, datatype, cluster, device):
    run=(str(dataset)+" "+str(sel)+" "+str(1024)+" "+str(cluster)+" "+str(datatype)+" "+str(device))
    print(run)
    cmd = './build/gpu_compressstore2 '+run
    #os.system(cmd)


if __name__ == '__main__':


    #bench 1: selectivity from 0.01 to 0.99 4% increments, datasize 1gib, all datasets, all datatypes

    #all selectivites
    device = "quadro"
    sel = 0
    dataset = 0

    max_cluster = 64

    for k in range(1,8,1):
        datatype = k


        for f in range(1, 100, 4):
            sel=f/100


            for i in range(0, 3, 1):
                dataset = i
                c = 1
                if dataset == 2:
                    while(c < max_cluster):
                        run_sel(sel, dataset, datatype, c,device)
                        c=c*2
                else:
                    run_sel(sel, dataset, datatype, 0,device)
                # print(dataset)

    gc.collect()
