import sys
import glob, os
import numpy as np
import math
import statistics
import scipy.integrate as integrate
import csv
import json
from collections import OrderedDict

folder = sys.argv[1]
r = int(sys.argv[2]) # resolution
w = int(sys.argv[3]) # window size
step = int(sys.argv[4]) # step size

def gaussian(x, sigma, mu):
    alpha = 2 * sigma * sigma
    return 1./(sigma * math.sqrt(2*math.pi))*np.exp(-np.power((x - mu) / sigma, 2.) / 2)

seg = [float(e) / r for e in range(-r, r + 1)]
l = integrate.quad(gaussian, -1, 1, args = (0.25, 0))[0]
# construct the quantization bins
bins = np.array([integrate.quad(gaussian, -1, s, args = (0.25, 0))[0] / l * 2 - 1 for s in seg])
sym2mid = {}
for idx in range(1, len(bins)):
    sym2mid[idx] = (bins[idx - 1] + bins[idx]) / 2

documents = []
components = ['W', 'X', 'Y', 'Z']
os.chdir(folder + "/W")
for filename in glob.glob('*.csv'):
    documents.append(filename)

for filename in documents:
    fn = os.path.splitext(filename)[0]
    result = OrderedDict()
    for component in components:
        result[component] = OrderedDict()
        # print(component)
        with open('../' + component + '/' + filename, mode = 'r') as file:
            sensors = csv.reader(file)
            sensorwords = []
            for idx, strs in enumerate(sensors):
                result[component][idx] = OrderedDict()
                pos = [float(f) for f in strs]
                result[component][idx]['avg'] = statistics.mean(pos)   # average of original data
                result[component][idx]['std'] = statistics.stdev(pos)  # standard deviation of original data
                # normalize to [-1, 1]
                drange = max(pos) - min(pos)
                pos = [(f - min(pos)) / drange * 2 - 1 if drange != 0 else 0.0 for f in pos] # direct current
                pos = np.array(pos)
                digi = np.digitize(pos, bins, True)
                digi = [1 if f == 0 else f for f in digi] # value 0 belongs to the 1st band
                result[component][idx]['winsymb'] = OrderedDict()  # map for storing symbolic quantized window descriptor
                result[component][idx]['winavg'] = OrderedDict()   # map for storing average quantized amplitude
                for i in range(0, len(digi) - w + 1, step):
                    result[component][idx]['winsymb'][i] = str(digi[i : i + w]) # average quantized amplitude
                    #symbo = np.digitize(statistics.mean(pos[i : i + w]), bins, True)
                    #symbo = 1 if symbo == 0 else symbo
                    # result[component][idx]['winavg'][i] = statistics.mean(digi[i : i + w])    # average quantized amplitude
                    result[component][idx]['winavg'][i] = statistics.mean([sym2mid[digi[sym]] for sym in range(i, i + w)])
    with open("../" + str(fn) + ".wrd","w") as f:
        def convert(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError
        json.dump(result,f,indent=2, default=convert)