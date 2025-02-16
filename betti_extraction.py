
import networkx as nx
import pyflagser
import pickle
import numpy as np


def print_stat(train_acc, test_acc,pre,rec,f1):
    argmax=np.argmax(train_acc)
    best_result=test_acc[argmax]
    train_acc=np.max(train_acc)
    test_acc=np.max(test_acc)
    print(f'Train accuracy = {train_acc*100:.2f}%,Test Accuracy = {test_acc*100:.2f}%,precision = {np.max(pre)*100:.2f}%,Recall = {np.max(rec)*100:.2f}%,f1-score = {np.max(f1)*100:.2f}%\n')
    return test_acc

def Average(lst):
    return sum(lst) / len(lst)


def Topo_Fe_TimeSeries_MP(TS_voltage, TS_branchFlow, F_voltage, F_Flow,N,E):
    betti_0 = []

    for k in range(len(TS_voltage)):
        fec = []
        Voltage = TS_voltage[k]
        # Compute AverageVoltage using NumPy (vectorized operation)
        AverageVoltage = np.array([Average(y) for y in Voltage])
        # Extract first column of TS_branchFlow using NumPy for efficiency
        BranchFlow = np.array([bf[0] for bf in TS_branchFlow[k]])
        for p in range(len(F_voltage)):
            # Precompute active nodes based on threshold F_voltage[p]
            Active_node_v = np.where(AverageVoltage > F_voltage[p])[0]
            for q in range(len(F_Flow)):
                if Active_node_v.size == 0:
                    fec.append(0)
                    continue

                # Create directed graph
                G = nx.DiGraph()
                G.add_nodes_from(Active_node_v)
                # Find edges where branch flow exceeds threshold F_Flow[q]
                indices = np.where(BranchFlow > F_Flow[q])[0]
                edges_to_add = [(int(N.index(E[s][0])), int(N.index(E[s][1]))) for s in indices]

                # Filter edges to include only active nodes
                edges_to_add = [(a, b) for a, b in edges_to_add if a in Active_node_v and b in Active_node_v]
                G.add_edges_from(edges_to_add)

                Adj = nx.to_numpy_array(G)
                my_flag = pyflagser.flagser_unweighted(Adj, min_dimension=0, max_dimension=2, directed=False, coeff=2)
                fec.append(my_flag["betti"][0])

        betti_0.append(fec)

    return betti_0
