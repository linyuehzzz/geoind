import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from gurobipy import Model, GRB, LinExpr, quicksum
import numpy as np
import csv


def get_nearest(src_points, candidates, k_neighbors=11):
    """
    Find nearest neighbors for all source points from a set of candidate points
    modified from: https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    """
    
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='euclidean')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Return indices and distances
    return indices, distances

# read data
gdf1 = gpd.read_file('data/franklin/franklin_sample.shp')
gdf2 = gpd.read_file('data/franklin/franklin_households_all.shp')
in_pts = [(x,y) for x,y in zip(gdf1.geometry.x , gdf1.geometry.y)]
qry_pts =  [(x,y) for x,y in zip(gdf2.geometry.x , gdf2.geometry.y)]

# prepare model inputs
k_neighbors_all = [11, 21, 31]
eps_all = [0.1, 0.01, 0.001, 0.0001]
M = len(gdf1)

with open('data/franklin/sols/franklin_runtime.csv', 'w', newline='') as fw:
    fw.write('eps,k,runtime\n')
    fw.flush()

    for eps in eps_all:
        for k_neighbors in k_neighbors_all:
            # generate masking area
            X, _ = get_nearest(in_pts, qry_pts, k_neighbors)
            with open('data/franklin/sols/franklin_prob_eps' + str(eps) + "_k" + str(k_neighbors-1) + '.csv', 'w', newline='') as f1:
                wr1 = csv.writer(f1)
                with open('data/franklin/sols/franklin_prob_obj_eps' + str(eps) + "_k" + str(k_neighbors-1) + '.csv', 'w', newline='') as f2:
                    wr2 = csv.writer(f2)

                    time = 0
                    for k in range(M):
                        # initialize model
                        m = Model('td')

                        # add objective function
                        obj = LinExpr()

                        # add decision variables and objective function
                        p = {}     ## decision vairable
                        d = np.zeros((k_neighbors, k_neighbors))
                        for i in range(k_neighbors):
                            for j in range(k_neighbors):
                                if j != i:
                                # objective
                                    d[i, j] = gdf2.iloc[X[k, i]].geometry.distance(gdf2.iloc[X[k, j]].geometry)
                                    p[i, j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="p_%d_%d"%(i, j))
                                    obj += p[i, j] * d[i, j]

                        # add constraints
                        for i1 in range(k_neighbors):
                            for j in range(k_neighbors):
                                for i2 in range(k_neighbors):
                                    if j != i1 and j != i2:
                                        m.addConstr(p[i1, j] <= np.exp(eps * d[i1, i2]) * p[i2, j])
                        for i in range(k_neighbors):
                            tmp = [j for j in range(k_neighbors) if j != i]
                            m.addConstr(quicksum(p[i, j] for j in tmp) == 1)
                        
                        m.setObjective(obj, GRB.MINIMIZE)

                        m.update()
                        m.optimize()

                        try:
                            time += m.Runtime
                            # write prob values
                            prob_values = [k]
                            var_values = [var.X for var in m.getVars() if 'p_0_' == str(var.VarName[0:4])]
                            prob_values.extend(var_values)
                            wr1.writerow(prob_values)

                            # write objective values
                            obj = m.getObjective().getValue()
                            wr2.writerow([k, obj])
                            print(k)
                        except:
                            pass # doing nothing on exception
                    fw.write(str(eps) + ',' + str(k_neighbors-1) + ',' + str(time) + '\n')
                    fw.flush()