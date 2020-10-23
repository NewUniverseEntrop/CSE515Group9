import numpy as np
import random

def randomCentroids(samples,k):  # pick k centroids from samples randomly strategy 1
    centeroids = []
    size = len(samples)
    for i in range(0, k):
        n = random.randint(0, size - 1)
        centeroids.append(samples[n])
    return centeroids

def pickCentroidsMethod2(samples,k): # pick k centroids using strategy 2
    centeroids = []
    size = len(samples)
    firstCentroid = random.randint(0, size - 1)  #pick first centroid randomly
    # print(firstCentroid, samples[firstCentroid], len(samples))
    centeroids.append(samples[firstCentroid])
    for i in range(1,k):
        index = pickFarthestCentroid(centeroids,samples) # pick sample whose avg distance from 0...i-1 centroid is max
        # print(index,samples[index],len(samples))
        centeroids.append(samples[index])
    return centeroids

def ifSampleisCenteroid(centeroids,sample): # ignore if sample itself is centroid
    for centeroid in centeroids:
        if all(centeroid == sample):
            return bool(1)
    return bool(0)
def pickFarthestCentroid(centeroids,samples):  # pick sample whose avg distance from 0...i-1 centroid is max
    new_samples = []
    for sample in samples:
        if not ifSampleisCenteroid(centeroids,sample):
            new_samples.append(sample)
    distances = []
    for sample in new_samples:
        distance = 0
        for centeroid in centeroids:
            norm = np.linalg.norm(centeroid - sample)
            distance += norm
        distances.append(distance/len(centeroids))
    return distances.index(max(distances))

def resetMembership(membershipMap,fileMembershipMap,k):
    for i in range(0,k):
        membershipMap[i] = []
        fileMembershipMap[i] = []


def classifySample(sample,centeroids):  #function to classify sample to nearest centroid
    distances = []
    for centeroid in centeroids:
        norm = np.linalg.norm(centeroid-sample)
        distances.append(norm)
    i = distances.index(min(distances))
    return i

def recomputeCentroids(samples,centeroids,membershipMap,filemembershipMap,k):  #computing new centroid after classifying samples to new centroid

    new_centeroids = []
    resetMembership(membershipMap,filemembershipMap,k)
    for index,sample in enumerate(samples):
        nearestCenteroid = classifySample(sample,centeroids)
        membershipMap[nearestCenteroid].append(sample)
        filemembershipMap[nearestCenteroid].append(index)
    for i in range(0,k):
        members = membershipMap[i]
        if len(members) == 0:  #no sample belong to this cluster
            mean = centeroids[i]
        else:
            mean = np.mean(members,axis=0)  # takinng mean of all samples to compute new centroid
        new_centeroids.append(mean)
    return new_centeroids

def costFunction(centeroids,membershipMap):  # objective function calculation
    variance = 0
    for i in membershipMap:
        centeroid = centeroids[i]
        members = membershipMap[i]
        for member in members:
            norm = np.linalg.norm(centeroid-member)
            variance = variance + norm*norm
    return variance

def runKmeanClustering(samples,k,strategy=1):
    membershipMap = {}
    fileMembershipMap = {}
    centeroids = []
    if strategy == 1:
        centeroids = randomCentroids(samples,k)  #strategy 1
    else:
        centeroids = pickCentroidsMethod2(samples,k)  # strategy 2

    resetMembership(membershipMap,fileMembershipMap,k)
    for index,sample in enumerate(samples):  # classifying each sample to nearest centroid
        nearestCenteroid = classifySample(sample, centeroids)
        membershipMap[nearestCenteroid].append(sample)
        fileMembershipMap[nearestCenteroid].append(index)

    end = bool(0)
    cost = costFunction(centeroids,membershipMap)      # initial cost of objective function
    while not end and cost!=0 :
        new_centeroids = recomputeCentroids(samples,centeroids,membershipMap,fileMembershipMap,k)  #recomputing centroids iteratively
        new_cost = costFunction(new_centeroids,membershipMap)
        delta_cost = (cost-new_cost)*100/cost  # relative change in cost in percent
        if(delta_cost < 0.1):  # convergence condition until there is not change in centroids
            end = bool(1)
        cost = new_cost
        centeroids = new_centeroids

    # print(membershipMap)
    print(fileMembershipMap)
    # print(np.array(centeroids))
    return fileMembershipMap

# samples =  np.random.randint(low=0,high=10,size=(3,3))
# print(samples)
# samples = np.array([[1,1,1],[2,2,2],[5,5,5],[6,6,6],[10,9,8],[11,9,9]])
# print(samples)
# runKmeanClustering(samples,3)
