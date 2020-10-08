import numpy as np

def editdist(series1, series2, avg1, avg2, std1, std2):
    channel = len(series1)
    m, n = len(series1[0]), len(series2[0])
    winsz = len(series1[0][0])
    table = [[0] * (n + 1) for _ in range(m + 1)]
    mid = 4.5
    for i in range(1, m + 1):
        #table[i][0] = table[i - 1][0] + sum([sum([abs(series1[ch][i - 1][p] - mid) for p in range(winsz)]) * std1[ch] for ch in range(channel)])
        table[i][0] = table[i - 1][0] + sum([std1[ch] for ch in range(channel)])
    for j in range(1, n + 1):
        #table[0][j] = table[0][j - 1] + sum([sum([abs(series2[ch][j - 1][p] - mid) for p in range(winsz)]) * std2[ch] for ch in range(channel)])
        table[0][j] = table[0][j - 1] + sum([std2[ch] for ch in range(channel)])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pass
            #d1 = table[i - 1][j] + sum([sum([abs(series1[ch][i - 1][p] - mid) for p in range(winsz)]) * std1[ch] for ch in range(channel)])
            #d2 = table[i][j - 1] + sum([sum([abs(series2[ch][j - 1][p] - mid) for p in range(winsz)]) * std2[ch] for ch in range(channel)])
            d1 = table[i - 1][j] + sum([std1[ch] for ch in range(channel)])
            d2 = table[i][j - 1] + sum([std2[ch] for ch in range(channel)])
            #print(np.corrcoef([2,3,4],[5,4,3]))
            #d3 = table[i - 1][j - 1] + sum([sum([abs(series1[ch][i - 1][p] - series2[ch][j - 1][p]) for p in range(winsz)]) * max(std1[ch], std2[ch]) for ch in range(channel)])
            #print([series1[ch][i - 1] for ch in range(channel)])
            d3 = table[i - 1][j - 1]
            for ch in range(channel):
                f1 = max(series1[ch][i - 1]) == min(series1[ch][i - 1])
                f2 = max(series2[ch][j - 1]) == min(series2[ch][j - 1])
                if f1 and f2:
                    pass
                elif f1 or f2:
                    d3 += (std1[ch] + std2[ch])
                else:
                    d3 += (1 - np.corrcoef(series1[ch][i - 1], series2[ch][j - 1])[0][1]) * (std1[ch] + std2[ch])
            table[i][j] = min(d1, d2, d3)
    return table[-1][-1]

def dtw(series1, series2):
    channel = len(series1)
    m, n = len(series1[0]), len(series2[0])
    table = [[0] * (n + 1) for _ in range(m + 1)]
    mid = 4.5
    for i in range(1, m + 1):
        table[i][0] = table[i - 1][0] + sum([abs(series1[ch][i - 1] - mid) for ch in range(channel)])
    for j in range(1, n + 1):
        table[0][j] = table[0][j - 1] + sum([abs(series2[ch][j - 1] - mid) for ch in range(channel)])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d1 = table[i - 1][j] + sum([abs(series1[ch][i - 1] - mid) for ch in range(channel)])
            d2 = table[i][j - 1] + sum([abs(series2[ch][j - 1] - mid) for ch in range(channel)])
            d3 = table[i - 1][j - 1] + sum([abs(series1[ch][i - 1] - series2[ch][j - 1]) for ch in range(channel)])
            table[i][j] = min(d1, d2, d3)
    return table[-1][-1]