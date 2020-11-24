## Group Members
* Haowen Fan hfan24@asu.edu
* Qiang Fu qiangfu3@asu.edu
* Ravi Pilla ravikiran14@asu.edu
* Manuha Vancha mvancha@asu.edu
* Xin Xu xinxu11@asu.edu
* Zhiheng Zhang zzhan339@asu.edu

## Phase 3
### Task 1
### python gesturedisc.py folder  tfidf dtw 3 svd
### python task1.py folder 3 10 1,3,5


### Task 2
[TBD] Manuha and Ravi
### python ppr_classifier folder


### Task 3: Multi-dimensional index structures and nearest neighbor search task:
#### Parameters:
* path
* q: query gesture
* k: number of hash functions per layer
* L: number of layers
* t: results requested
* vector model (optional): tf (default), tfidf
```
python task3.py ../3_class_gesture_data 249 5 10 10
```

### Task 4
#### Parameters: probabilistic relevance feedback
* path
* relevant: objects user marked as relevant
* irrelevant: objects user marked as irrelevant
* neutral: objects with no opinion provided
```
python task4.py ../3_class_gesture_data 1,2 255,549 10,11,12
```

### Task 5
#### Parameters: classifier-based relevance feedback
* path
* relevant: objects user marked as relevant
* irrelevant: objects user marked as irrelevant
* neutral: objects with no opinion provided
```
python task5.py ../3_class_gesture_data 1,2 255,549 10,11,12
```

## Phase 2
### Task 0a
#### Parameters:
* path
* resolution
* window size
* step size
```
python gesturewords.py ../3_class_gesture_data/ 4 3 3
```
### Task 0b
#### Parameters:
* path
```
python gesturevectors.py ../3_class_gesture_data
```
### Task 1
#### Parameters:
* path
* vector model: tf, tfidf
* k: top-k latent semantics / topics
* option: pca, svd, nmf, lda
```
python gesturelatent.py ../3_class_gesture_data 'tfidf' 20 'pca'
```
### Task 2
#### Parameters:
* path
* query gesture
* vector model: tf, tfidf; ignored if option is `ed' or `dtw'
* option: dotp, pca, svd, nmf, lda, ed, dtw
```
python gesturesimilar.py ../3_class_gesture_data 249 'tfidf' 'pca'
```
### Tasks 3 & 4
#### Parameters:
* path
* vector model: tf, tfidf; ignored if option is `ed' or `dtw'
* option: dotp, pca, svd, nmf, lda, ed, dtw
* topp:  latent semantics to be extracted from the gestures; or cluster number of k-means and spectral clustering
* grouping: grouping strategy, can be chosen from svd, nmf, kmeans, spectral
```
python gesturedisc.py . 'tfidf' 'pca' 3 'kmeans'
```
