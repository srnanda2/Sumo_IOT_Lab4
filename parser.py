import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import sys
import queue
import random
import json
from datetime import datetime
import os
import numpy as np
import argparse


net_path = "osm.net.xml"
fcd_path = "fcd.xml"
queue_path = "queue.xml"
tmp_path = "tmp.txt"
data_path = "data.txt"
label_path = "label.txt"

# Load the sumo network from osm.net file to networkx graph structure
def get_graph(net_path):
    """
    Read sumo network from net_path
    Return a networkx graph and a lookup tabel (dictionary) to map internal road (auxilary road will appear in fcd file) to real road
    """
    G = nx.Graph()
    countdown = 5
    internal = {}
    for _, elem in ET.iterparse(net_path):
        if elem.tag == "edge":
            fm = elem.get("from")
            to = elem.get("to")
            edge_id = elem.get("id")
            if edge_id[0] == '-':
                edge_id = edge_id[1:]
                
            if G.has_edge(fm,to):
                G[fm][to]["edge_id"].append(edge_id)
            
            elif fm is not None and to is not None:
                num_lane = len( elem )
                length = float( elem[0].get("length") )
                G.add_edge(fm,to, edge_id=[edge_id], num_lane=num_lane, length=length)

        if elem.tag == "connection":
            fm = elem.get("from")
            to = elem.get("to")
            if fm[0] == ':':
                internal[fm] = to

    return G, internal

# random car position datas in the fcd file. Then for each position, use graph traversal algorithm to generating subgraph nearby. and return all subgraph generated. And save
# save information used to determine if each subgraph have jam in <tmp path>
def generating_data(fcd_path, tmp_path, size, subG_range, internal):
    """
    Return: SubGs:  subgraphs generated
            tsps:   sampled timestep cooresponding to each subgraph in subGs;
            nodes_depth:    a list of dictionary that contain nodes for each corresponding subgraph
    """
    open(tmp_path, 'w').close()
    car_line_size = 156
    random.seed(datetime.now())
    num_car = os.path.getsize(fcd_path)/car_line_size
    idx = sorted(random.sample(range(int(num_car)), size))
    acc = 0
    i = 0
    subGs = []
    tsps = []
    nodes_dicts = []

    for _, elem in ET.iterparse(fcd_path):
        if elem.tag == "timestep":
            
            while i < len(idx) and idx[i] < acc + len(elem):
                vh = elem[idx[i] - acc]
                rd = vh.get("lane")
                rd = conform_road_format(rd, internal)

                res = find_edge(G, rd)
                if res is None:
                    i += 1
                    continue
    
                u, v = res
                subG, node_depth = get_subgraph_by_DJK(G, u, subG_range)
                subGs.append(subG)
                nodes_dicts.append(node_depth)
                tsps.append(float(elem.get("time")))

                with open(tmp_path, 'a') as f:
                    f.write("{}".format(elem.get("time")))
                    s1 = set([item  for lst in [rd for rd in [d["edge_id"] for d in G[u].values()]] for item in lst])
                    s2 = set([item  for lst in [rd for rd in [d["edge_id"] for d in G[v].values()]] for item in lst])
                    rds = s1.union(s2)
                    for item in rds:
                        f.write(" {}".format(item))
                    f.write("\n")
                i += 1

            acc += len(elem)
            elem.clear()
            if i >= len(idx):
                break

    return subGs, tsps, nodes_dicts

# generating the adjacency matrix for each subgraph and write them to <save_path>
def load_vehicle_data_to_road_network(fcd_path, save_path, G, subGs, tsps, nodes_dicts, size):
    open(save_path, 'w').close()
    lanes = [[item for sublist in [G[u][v]["edge_id"] for u,v in subGs[i].edges()] for item in sublist] for i in range(len(subGs))]
    count_net = [{l:0 for l in lane} for lane in lanes]
    acc = 0
    
    for _, elem in ET.iterparse(fcd_path):
        if elem.tag == "timestep":
            i = acc
            for tsp in tsps[acc:]:
                if float(elem.get("time")) > tsp:
                    acc += 1
                    adj_raw = get_adjacency_matrix(subGs[i], nodes_dicts[i], size)
                    adj = get_adjacency_matrix_custom_data(subGs[i], nodes_dicts[i], count_net[i], size)
                        
                    with open(save_path, 'a') as f:
                        for i in range(len(adj)):
                            f.write("{} {} ".format(adj_raw[i], adj[i]))
                        f.write("\n")

                    i += 1
                    continue
                
                if float(elem.get("time")) == tsp:
                    for veh in elem:
                        lane_id = veh.get("lane")[:-2]
                        if lane_id in lanes[i]:
                            count_net[i][lane_id] += 1
                i += 1
                            
            if float(elem.get("time")) > tsps[-1]+1:
                break
            elem.clear()

# helper function to read the temporary file to determine the traffic hotspot.
def read_label_generating_data(src):
    tsps = []
    rds = []
    with open(src) as f:
        for line in f:
            strlist = line.split()
            tsp = int(float(strlist[0]))
            tsps.append(tsp)
            rds.append([])
            for rd in strlist[1:]:
                rds[-1].append(rd)
    return tsps, rds

# helper function of dijkstra, find the node cloest to source within the set sqt
def find_min_dist(dist, spt):
    min_v = 2147483648
    
    for k, v in dist.items():
        if min_v > v and k not in spt:
            min_k = k
            min_v = v
    return k, v

# graph traversal algorithm, traverse the nodes closest to the source first
def dijkstra(G, source, dist_limit):
    dist = {}
    dist[source] = 0
    spt = set()
    for _ in range(G.number_of_nodes()):
        node, d  = find_min_dist(dist, spt)
        
        if d > dist_limit:
            break
        
        spt.add(node)
        neighbor = G[node]
        for n in neighbor:
            if n not in dist:
                dist[n] = 2147483648
            l = neighbor[n]["length"]
            if l < dist[n]:
                dist[n] = l
    return dist

# generate the subgraph by D
def get_subgraph_by_DJK(G, source, dist_limit):
    djk = dijkstra(G, source, dist_limit)
    return G.subgraph(list(djk.keys())), djk 

# Broad First Search
def BFS(G, source, depth_limit):
    d = {}
    front = queue.Queue()
    front.put((source, 0))
    while not front.empty():
        node, depth = front.get()
        d[node] = depth
        if depth < depth_limit:
            neighbor = G.neighbors(node)
            for n in neighbor:
                if n not in d:
                    front.put((n, depth+1))
    
    return d

# (Alternative) generate the subgraph by BFS
def get_subgraph_by_BFS(G, source, depth_limit):
    bfs = BFS(G, source, depth_limit)
    return G.subgraph(list(bfs.keys())), bfs 

# find the connecting nodes of certain edge
def find_edge(G, edge_id):
    for u, v, e in G.edges(data=True):
        if edge_id in e["edge_id"]:
            return (u, v)
    print(edge_id)

# fix the format of road in sumo network
def conform_road_format(edge_id, internal):
    if edge_id[-2] == '_':
        edge_id = edge_id[:-2]
    if edge_id[0] == ':':
        edge_id = internal.get(edge_id, edge_id)
    if edge_id[0] == '-':
        edge_id = edge_id[1:]
    if edge_id[-2] == '_':
        edge_id = edge_id[:-2]
    return edge_id

# get the adjacency matrix where each nonzero entry represents an edge and has the value of road length
def get_adjacency_matrix(G, nodes_depth, size=-1):
    if size == -1:
        size = len(node_depth)
    depth_nodes = {}
    for key, val in nodes_depth.items():
        if key not in depth_nodes:
            depth_nodes[key] = []
        depth_nodes[key].append(val)
        
    idx = 0
    nodes = ["" for _ in range(len(nodes_depth))]
    for k in depth_nodes:
        for _ in range(len(depth_nodes[k])):
            nodes_depth[k] = idx
            nodes[idx] = k
            idx += 1
    
    ajm = [0 for _ in range(size*size)]
    for i in range(len(nodes)):
        neighbor = G[nodes[i]]
        for n in neighbor:
            idx = i*size + nodes_depth[n]
            if idx < size*size:
                ajm[idx] = neighbor[n]["length"]
    
    return ajm

# get the adjacency matrix where each nonzero entry represents an edge and has the value of in the data
def get_adjacency_matrix_custom_data(G, nodes_depth, data, size=-1):
    if size == -1:
        size = len(node_depth)
    depth_nodes = {}
    for key, val in nodes_depth.items():
        if key not in depth_nodes:
            depth_nodes[key] = []
        depth_nodes[key].append(val)
        
    idx = 0
    nodes = ["" for _ in range(len(nodes_depth))]
    for k in depth_nodes:
        for _ in range(len(depth_nodes[k])):
            nodes_depth[k] = idx
            nodes[idx] = k
            idx += 1
    
    ajm = [0 for _ in range(size*size)]
    for i in range(len(nodes)):
        neighbor = G[nodes[i]]
        for n in neighbor:
            idx = i*size + nodes_depth[n]
            if idx < size*size:
                ajm[idx] = data[G[nodes[i]][n]["edge_id"][0]]
    
    return ajm

#depreciated O(N^2) algorithm
def determine_traffic_hotspot(src, tsp, lanes, internal, time_range=1, num_car_threshold=1, waiting_time_threshold=60, jam_threshold=0):
    trange = range(tsp, tsp+time_range)
    num_jam = {}
    for lane in lanes:
        num_jam[lane] = 0
    
    for _, elem in ET.iterparse(src):
        if elem.tag == "data":
            if float(elem.get("timestep")) >= tsp:
                for lane in elem[0]:
                    lane_id = lane.get("id")
                    lane_id = conform_road_format(lane_id, internal)
                    if lane_id in lanes:
                        if float(lane.get("queueing_time")) > waiting_time_threshold and \
                        float(lane.get("queueing_length")) > num_car_threshold:
                            num_jam[lane_id] += 1
            if float(elem.get("timestep")) >= tsp+time_range:
                break
            elem.clear()
            
    if_jam = {}
    for k,v in num_jam.items():
        if v > jam_threshold:
            if_jam[k] = True
        else:
            if_jam[k] = False
            
    return if_jam

def determine_traffic_hotspot_group(src, save_src, tsps, lanes, internal, time_offset=0, time_range=1, num_car_threshold=0, waiting_time_threshold=0, jam_threshold=0):
    num_jam = [0 for _ in lanes]
    acc = 0
    
    for _, elem in ET.iterparse(src):
        if elem.tag == "data":
            i = acc
            for tsp in tsps[acc:]:
                if float(elem.get("timestep")) >= tsp+time_offset+time_range:
                    acc += 1
                    i += 1
                    continue
                
                if float(elem.get("timestep")) >= tsp+time_offset:
                    for lane in elem[0]:
                        lane_id = lane.get("id")
                        lane_id = conform_road_format(lane_id, internal)
                        if lane_id in lanes[i]:
                            if float(lane.get("queueing_time")) > waiting_time_threshold and \
                            float(lane.get("queueing_length")) > num_car_threshold:
                                num_jam[i] += 1
                i += 1
                            
            if float(elem.get("timestep")) >= tsps[-1]+time_offset+time_range:
                break
            elem.clear()
    if_jam = [0 for _ in lanes]
    for i in range(len(num_jam)):
        if num_jam[i] > jam_threshold:
            if_jam[i] = 1
        else:
            if_jam[i] = 0
            
    with open(save_src, 'w') as f:
        for lab in if_jam:
            f.write("{}\n".format(lab))
            
    return if_jam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcd", type=str, default="fcd.xml", help="path to the fcd file")
    parser.add_argument("--queue", type=str, default="queue.xml", help="path to the queue file")
    parser.add_argument("--network", type=str, default="osm.net.xml", help="path to the sumo network file ")

    parser.add_argument("--size", default=8000, type=int, help="number of data point to generate")
    parser.add_argument("--matrix_size", default=10, type=int, help="size of ajacency matrix of each data point")
    parser.add_argument("--subG_range", default=300, type=int, help="range (in meters) of road network that would be considered as feature of prediction")
    parser.add_argument("--prediction_time", default=1, type=int, help="we predict whether traffic hotspot appears after <prediction_time> time step")
    parser.add_argument("--hotspot_time_interval", default=1, type=int, help="length of time window that would be used to determine traffic hotspot")
    parser.add_argument("--waiting_car_threshold", default=0, type=int, help="minimum length of cars waiting that a road would be considered as a traffic jam")
    parser.add_argument("--waiting_time_threshold", default=0, type=float, help="minimum waiting time of cars where a road be considered as a traffic jam")
    parser.add_argument("--jam_threshold", default=0.5, type=float, help="if within a time range, more than (jam_threshold) ratio of timestep traffic jam \
                                                                        occur on a road then this road would be considered as traffic hotspot")

    args = parser.parse_args()

    fcd_path = args.fcd
    queue_path = args.queue
    net_path = args.network

    size = args.size
    matrix_size = args.matrix_size
    subG_range = args.subG_range
    t_offset = args.prediction_time
    time_interval = args.hotspot_time_interval
    waiting_car_threshold = args.waiting_car_threshold
    waiting_time_threshold = args.waiting_time_threshold
    jam_threshold = args.jam_threshold
    jam_threshold = max(min(1, jam_threshold), 0)

    G, internal = get_graph(net_path)

    subGs, tsps, nodes_dicts = generating_data(fcd_path, tmp_path, size, subG_range, internal)

    load_vehicle_data_to_road_network(fcd_path, data_path, G, subGs, tsps, nodes_dicts, matrix_size)

    tsps, rds = read_label_generating_data(tmp_path)

    determine_traffic_hotspot_group(queue_path, label_path, tsps, rds, internal, time_offset=t_offset, time_range=time_interval, num_car_threshold=waiting_car_threshold, waiting_time_threshold=waiting_time_threshold, jam_threshold=jam_threshold)
    
    os.remove(tmp_path)

