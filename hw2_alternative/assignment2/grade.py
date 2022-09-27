import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT_DIR, 'data', 'assignment2', 'BBW', 'results')
STD_DIR = os.path.join(ROOT_DIR, 'data', 'assignment2', 'BBW', 'std')

testcases = ['cube_voxel', 'frame', 'bunny', 'spot', 'tyra']

for testcase in testcases:
    exist = True
    try:
        fp_res = open(os.path.join(RESULT_DIR, f'{testcase}_bbw_weights.txt'), 'r')
    except FileNotFoundError:
        exist = False
    if not exist:
        print('{:15s}: not found'.format(testcase))
        continue
    fp_std = open(os.path.join(STD_DIR, f'{testcase}_bbw_weights.txt'), 'r')

    # read your results
    data_res = fp_res.readlines()
    n_res = len(data_res)
    m_res = len(data_res[0].split())
    data_std = fp_std.readlines()
    n_std = len(data_std)
    m_std = len(data_std[0].split())
    fp_res.close()
    fp_std.close()

    if n_res != n_std or m_res != m_std:
        print('{:15s}: result has a wrong number of elements'.format(testcase))
    else:
        error = 0.0
        for i in range(n_std):
            linedata_res = data_res[i].split()
            linedata_std = data_std[i].split()
            if 'nan' in linedata_res:
                error = np.inf
                break
            for j in range(m_std):
                a_res = float(linedata_res[j])
                a_std = float(linedata_std[j])
                cur_error = abs(a_res - a_std) / max(abs(a_std), 1)
                error = max(error, cur_error)
        if error < 1e-5:
            print('{:15s}: pass'.format(testcase))
        else:
            print('{:15s}: wrong answer'.format(testcase))

