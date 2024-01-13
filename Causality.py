import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
import networkx as nx
#import castle
# !pip3 install gcastle 
import warnings
warnings.filterwarnings("ignore")
# import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress, spearmanr
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

import matplotlib.pyplot as plt
# from mpl_toolkits import axes_grid1


df = pd.read_csv('SAMI_CBSA.csv') 
index = [
        "SAMI_COLLEGE_EDUCATION",
        "SAMI_MHP",
        "SAMI_ADHD",
        "SAMI_FI_CHILD",
        "SAMI_OBESITY_REAL",
        "SAMI_PHYSINACT_REAL", 
        
         ]

pc_dataset = np.array([df[i] for i in index]).T
                       

pc_Fisher = PC(alpha=.05)
pc_Fisher.learn(pc_dataset)
learned_graph = nx.DiGraph(pc_Fisher.causal_matrix)
spearman_table = df[index]
attribute = {}
for i in range(len(index)):
    for j in range(len(index)):
        if j!=i and pc_Fisher.causal_matrix[i,j]!=0:
            if spearman_table.corr('pearson').iloc[i,j]<0:
                attribute[(i,j)] = "b"
            else:attribute[(i,j)] = "r"
            # nx.set_edge_attributes(learned_graph,)

labels = []
for i in index: 
    if "SAMI" in i : labels.append(i[5:]); 
    else: labels.append(i)
MAPPING  = {k: v for k, v in zip(range(len(labels)), labels)}

learned_graph  = nx.relabel_nodes(learned_graph, MAPPING, copy=True)
# Plot the graph
plt.figure(figsize=(20,20))

layout = nx.circular_layout(learned_graph)
nx.draw(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    learned_graph, 
    pos=layout,
    with_labels=True,
    node_size=40000,
    node_color = "dimgray",
    font_size=20,
    font_color='white',
    arrowsize=35,
    # arrowstyle='->',
    edge_color=attribute.values(),
)
# plt.savefig("SAMI_CAUSAL.pdf")
plt.show()