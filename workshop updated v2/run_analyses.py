import sys, os
sys.path.append('models')
sys.path.append('modified_fancyimpute')
from greedymerge_synthetic import main as gms_main
from latest_synthetic import main as ls_main
from soft_impute_synthetic import main as sis_main
from greedymerge_image import main as gmi_main
from latest_image import main as li_main
import numpy as np


def processing_data(list_):
    l = len(list_)
    r90 = 0
    r100 = 9900
    for k in range(1, l):
        if list_[k-1]<0.9 and list_[k]>=0.9:
            r90 = 100*(k + 1)
            break
    for k in range(1, l):
        if list_[k-1]<1.0 and list_[k]==1.0:
            r100 = 100*(k + 1)
            break
    if list_[l-1]<1:
        r100 = 10000
    return r90, r100

latest_syn_accuracy_list = []
gm_syn_accuracy_list = []
image_accuracy_list = []
si_syn_accuracy_list = []
ibc = []
gm = []
si = []

hybrid90 = np.zeros((9, 9))
hybrid100 = np.zeros((9, 9))
random90 = np.zeros((9, 9))
random100 = np.zeros((9, 9))
if __name__ == "__main__":
    i = 0
    for u in np.arange(0.1, 1, 0.1):
        j = 0
        for r in np.arange(0.1, 1, 0.1):
            a, b = processing_data(ls_main(u, r, "hybrid", 32))
            hybrid90[i, j] = a
            hybrid100[i, j] = b
            a, b = processing_data(ls_main(u, r, "random", 32))
            random90[i, j] = a
            random100[i, j] = b

    np.save('processed data' + os.path.sep + "hybrid90", hybrid90)
    np.save('processed data' + os.path.sep + "hybrid100", hybrid100)
    np.save('processed data' + os.path.sep + "random90", random90)
    np.save('processed data' + os.path.sep + "random100", random100)




    i = 0
    gm = np.zeros(10, )
    ibc = np.zeros(10, )
    si = np.zeros(10, )
    for u in [0.4, 0.8]:
        for r in [1.0]:
            a, _ = processing_data(ls_main(u, r, "hybrid", 32))
            ibc[i] = a
            a, _ = processing_data(gms_main(u, r, 32))
            gm[i] = a
            a, _ = processing_data(sis_main(u, r, 32))
            si[i] = a
            i += 1
    np.save('processed data' + os.path.sep + 'latest90.npy', ibc)
    np.save('processed data' + os.path.sep + 'gm90.npy', gm)
    np.save('processed data' + os.path.sep + 'si90.npy', si)

    image_accuracy_list.append(gmi_main(strategy='active'))
    image_accuracy_list.append(li_main(strategy='hybrid'))
    image_accuracy_list.append(li_main(strategy='score'))
    image_accuracy_list.append(li_main(strategy='entropy'))
    image_accuracy_list.append(li_main(strategy='random'))
    image_accuracy_list.append(gmi_main(strategy='random'))

    np.save('processed data' + os.path.sep + "image_results", image_accuracy_list)



