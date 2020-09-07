import networkx as nx
import matplotlib.pyplot as plt

# 그래프 생성
G = nx.DiGraph()

G.add_nodes_from(['Strategic Program Manager, Customer Solutions (Contract)(66)', 'Strategic Program Manager, Customer Solutions (Contract)(89)', 'Contracts Manager(78)', 'LCS Project Manager(13192)'])

G.add_weighted_edges_from([('Strategic Program Manager, Customer Solutions (Contract)(66)', 'Strategic Program Manager, Customer Solutions (Contract)(89)', 0.125), ('Contracts Manager(78)', 'LCS Project Manager(13192)', 0.5)])

degree = nx.degree(G)
print(degree)

nx.draw(G,node_size=[500 + v[1]*500 for v in degree], with_labels=True)
plt.show()