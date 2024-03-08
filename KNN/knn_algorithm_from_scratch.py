import numpy as np
from collections import Counter

points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'red': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [3,9]

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNearestNeighbours:
    def __init__(self, k=3) -> None:
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

classifier = KNearestNeighbours()
classifier.fit(points)

print(classifier.predict(new_point))