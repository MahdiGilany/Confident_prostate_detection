from os.path import join as pjoin
import time
import pickle


def main():
    filename = 'BK_RF_P1_140_balance__20210203-175808'
    tic = time.time()
    with open(pjoin('../files', filename + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    print(time.time() - tic)


if __name__ == '__main__':
    main()
