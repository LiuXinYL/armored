import pandas as pd
import numpy as np







if __name__ == '__main__':

    a1 = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        [[100, 200, 300], [400, 500, 600], [700, 800, 900]],
    ])

    a2 = np.transpose(a1, (1, 0, 2))
    a3 = np.transpose(a1, (1, 2, 0))

    # print(a1[a1 >= 4])
    # print(np.arange(0.1, 1, 0.1))
    #
    # df = pd.DataFrame(a1)

    # df = pd.read_csv('./datas/hillbert/2缸_hillbert_垂直振动.csv')
    # df.to_csv('./datas/hillbert/2缸_hillbert_垂直振动.txt', index=False, sep='\t')
    print(a1.shape)
