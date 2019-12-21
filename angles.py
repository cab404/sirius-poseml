import numpy as np

def get_ang(pose):
    features = []
    for i in range(len(pose)):
        for j in range(len(pose)):
            if j != i:
                for k in range(len(pose)):
                    if k > i and k != j:
                        a = pose[i, :-1]
                        b = pose[j, :-1]
                        c = pose[k, :-1]
                        ba = a - b
                        bc = c - b
                        ang = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
                        features.append(ang)
    return features
