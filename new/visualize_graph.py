import matplotlib.pyplot as plt
import numpy as np
import networkx as ntwx

def show_graph(graph,
               label_values_of_nodes,
               label_colors_of_nodes,
               colors_of_edges='black',
               display_window_size=15,
               positions_of_nodes=None,
               cmap='jet'):
    """helps visualize the graph"""
    fig, ax = plt.subplots(figsize=(display_window_size, display_window_size))
    if positions_of_nodes is None:
        # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        positions_of_nodes = ntwx.spring_layout(graph,
                                                k=5/np.sqrt(graph.number_of_nodes()))
        # https://networkx.org/documentation/stable/reference/drawing.html
    ntwx.draw(
        graph,
        positions_of_nodes,
        with_labels=label_values_of_nodes, 
        labels=label_values_of_nodes, 
        node_color=label_colors_of_nodes, 
        ax=ax,
        cmap=cmap,
        edge_color=colors_of_edges)
    plt.close(fig)  # Close the figure to prevent multiple graph displays
    return fig  # Return the figure for potential further use or display
    
    
def plot_training_curves(train_losses, test_losses, grid=False):
    """shows training curves"""
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(np.log10(train_losses), label='train')
    ax1.plot(np.log10(test_losses), label='test')
    ax1.legend()
    if grid:
        ax1.grid()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(accs, label='acc')
    ax2.set(ylim=[0,1])
    ax2.legend()
    
    if grid:
        ax2.grid()
    
    plt.show()

