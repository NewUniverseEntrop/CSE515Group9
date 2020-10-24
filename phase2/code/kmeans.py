import numpy as nump
import random

def init1(points, k):
    centers = []
    size = len(points)
    for i in range(0, k):
        n = random.randint(0, size - 1)
        centers.append(points[n])
    return centers

def init2(points, k):
    centers = []
    size = len(points)
    firstCentroid = random.randint(0, size - 1)
    centers.append(points[firstCentroid])
    for i in range(1,k):
        index = getMostAwayCenter(centers, points)
        centers.append(points[index])
    return centers

def ifSampleisCenteroid(centeroids,sample):
    for centeroid in centeroids:
        if (centeroid == sample).all():
            return bool(1)
    return bool(0)
def getMostAwayCenter(centers, points):
    new_points = []
    for point in points:
        if not ifSampleisCenteroid(centers,point):
            new_points.append(point)
    distances = []
    for new_point in new_points:
        distance = 0
        for center in centers:
            norm = nump.linalg.norm(center - new_point)
            distance += norm
        distances.append(distance/len(centers))
    return distances.index(max(distances))

def resetAll(groupsMap, filegroupsMap, k):
    for i in range(0,k):
        groupsMap[i] = []
        filegroupsMap[i] = []


def assignGroup(sample, centeroids):
    distances = []
    for centeroid in centeroids:
        norm = nump.linalg.norm(centeroid - sample)
        distances.append(norm)
    i = distances.index(min(distances))
    return i

def regroup(points, centers, groupsMap, filegroupsMap, k):

    new_centers = []
    resetAll(groupsMap, filegroupsMap, k)
    for index,sample in enumerate(points):
        nearestCenter = assignGroup(sample, centers)
        groupsMap[nearestCenter].append(sample)
        filegroupsMap[nearestCenter].append(index)
    for i in range(0,k):
        members = groupsMap[i]
        if len(members) == 0:
            mean = centers[i]
        else:
            mean = nump.mean(members, axis=0)
        new_centers.append(mean)
    return new_centers

def costObjective(centers, groupsMap):  #cost
    total = 0
    for index in groupsMap:
        center = centers[index]
        members = groupsMap[index]
        for x in members:
            euclidena = nump.linalg.norm(center - x)
            total = total + euclidena*euclidena
    return total

def performClustering(points, k, strategy=2):
    groupsMap = {}
    filegroupsMap = {}
    centers = []
    if strategy == 1:
        centers = init1(points, k)  #init 1
    else:
        centers = init2(points, k)  # init 2

    resetAll(groupsMap, filegroupsMap, k)
    for index,sample in enumerate(points):
        nearestCenteroid = assignGroup(sample, centers)
        groupsMap[nearestCenteroid].append(sample)
        filegroupsMap[nearestCenteroid].append(index)

    end = bool(0)
    cost = costObjective(centers, groupsMap)
    while not end and cost!=0 :
        new_centers = regroup(points, centers, groupsMap, filegroupsMap, k)
        new_cost = costObjective(new_centers, groupsMap)
        delta_cost = (cost-new_cost)*100/cost
        if(delta_cost < 0.1):
            end = bool(1)
        cost = new_cost
        centers = new_centers

    # print(filegroupsMap)
    return filegroupsMap


# samples = nump.array([[1, 1, 1], [2, 2, 2], [5, 5, 5], [6, 6, 6], [10, 9, 8], [11, 9, 9]])
# print(samples)
# performClustering(samples, 3)
