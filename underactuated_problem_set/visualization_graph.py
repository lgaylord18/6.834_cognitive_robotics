import networkx as nx
import matplotlib.pyplot as plt

g = nx.DiGraph()
plt.ion()
K=50
for i in range(K):
	g.add_edge(i,1+i)
	nx.draw(g,with_labels=True)
	plt.draw()
	plt.pause(.000001)
	plt.gcf().clear()

while True:
	plt.pause(.5)

print(g.nodes(),'g')

