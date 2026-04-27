import osmnx as ox

def download_delhi_network():

    print("[OSM] Downloading Delhi road network...")

    G = ox.graph_from_place(
        "Delhi, India",
        network_type="drive"
    )

    ox.save_graphml(G, "data/raw/delhi_road_network.graphml")

    print("[OSM] Road network saved")


if __name__ == "__main__":
    download_delhi_network()