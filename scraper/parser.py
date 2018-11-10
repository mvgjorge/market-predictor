import csv
import sys
import numpy as np


if len(sys.argv) < 3:
    print('Usage: ./parser filePath percent')
    exit()
filename = sys.argv[1]
percent = sys.argv[2]

finalOutput = './input/tickerList.csv'

with open(filename) as file:
    reader = csv.reader(file)
    rows = list(reader)
    # sort them in descending order
    rows = np.array(rows)
    # print (rows[0])
    rows = rows[rows[:, 2].argsort()[::-1]]
    (n, d) = np.shape(rows)
    toGet = int(percent) * n // 100
    rows = rows[0:toGet, :]
    with open('./input/tickerList.csv', 'ab') as final:
        np.savetxt(final, rows, delimiter=",", fmt="%s")

''' op = None

with open('./input/tickerList.csv', 'r') as final:
    reader = csv.reader(final)
    rows = list(reader)
    # sort them in descending order
    rows = np.array(rows)
    print (np.array(rows[0]))
    rows = rows[rows[:, 2].argsort()[::-1]]
    (n, d) = np.shape(rows)
    toGet = int(percent) * n // 100
    rows = rows[0:toGet, :]
    op = rows

with open('./input/tickerList.csv', 'wb') as final:
    np.savetxt(final, op, delimiter=",", fmt="%s")
'''
