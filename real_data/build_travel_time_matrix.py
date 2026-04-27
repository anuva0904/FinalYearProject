import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx


def build_time_matrix():

    print("[OSM] Building travel time matrix")

    G = ox.load_graphml("data/raw/delhi_road_network.graphml")

    stops = pd.read_csv("data/processed/bus_stops_nodes.csv")

    nodes = stops["node_id"].tolist()

    N = len(nodes)

    tm = np.zeros((N, N))

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            try:
                length = nx.shortest_path_length(
                    G,
                    nodes[i],
                    nodes[j],
                    weight="length"
                )

                # convert meters to minutes
                speed_kmph = 30
                tm[i][j] = (length / 1000) / speed_kmph * 60

            except:
                tm[i][j] = 999

    np.save("data/processed/real_time_matrix.npy", tm)

    print("[OSM] Travel time matrix saved")


if __name__ == "__main__":
    build_time_matrix()
    