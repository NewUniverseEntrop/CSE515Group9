import sys
import glob, os
import numpy as np
import json
import ast
from collections import OrderedDict

folder = sys.argv[1]

os.chdir(folder)
docunum = 0
words = OrderedDict()

for filename in glob.glob('*.wrd'):
    fn = os.path.splitext(filename)[0]
    with open(filename) as json_file:
        data = json.load(json_file)
        words[fn] = data

filecnt = len(words)
tf = OrderedDict()
idf = OrderedDict()
# find the term frequency for each word in a file
for fn, filewords in words.items():
    for component, comwords in filewords.items():
        for sensor, sendata in comwords.items():
            # inverse of word count, for normalation of TF: window number x component * sensor
            den = float(1) / (len(sendata['winsymb']) * len(filewords) * len(comwords))
            for symbols in sendata['winsymb'].values():
                if (fn, component, sensor, symbols) not in tf:
                    tf[(fn, component, sensor, symbols)] = den
                else:
                    tf[(fn, component, sensor, symbols)] += den

# appearance in documents
for (fn, component, sensor, symbols) in tf.keys():
    if (component, sensor, symbols) not in idf:
        idf[(component, sensor, symbols)] = 1
    else:
        idf[(component, sensor, symbols)] += 1
# get the inverse for IDF
for k, v in idf.items():
    idf[k] = np.log(float(filecnt) / float(v))

# TF-IDF is TF x IDF
tfidf = OrderedDict()
for k, v in tf.items():
    tfidf[str(k)] = v * idf[(k[1], k[2], k[3])]

# make the keys strings, for dumping to JSON
for key in tf.keys():
    if type(key) is not str:
        try:
            tf[str(key)] = tf[key]
        except:
            try:
                tf[repr(key)] = tf[key]
            except:
                pass
        del tf[key]

with open("tf.txt","w") as f:
    json.dump(tf, f, indent=2)

with open("tfidf.txt","w") as f:
    json.dump(tfidf, f, indent=2)
