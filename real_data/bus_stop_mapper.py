import osmnx as ox
import pandas as pd


def map_bus_stops():

    print("[OSM] Mapping bus stops to nearest road nodes")

    G = ox.load_graphml("data/raw/delhi_road_network.graphml")

    stops = pd.read_csv("data/raw/bus_stops.csv")

    nodes = []

    for _, row in stops.iterrows():

        node = ox.distance.nearest_nodes(
            G,
            row["lon"],
            row["lat"]
        )

        nodes.append(node)

    stops["node_id"] = nodes

    stops.to_csv("data/processed/bus_stops_nodes.csv", index=False)

    print("[OSM] Bus stops mapped to road nodes")


if __name__ == "__main__":
    map_bus_stops()
    