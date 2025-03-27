import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import networkx as nx
import matplotlib.colors as mcolors


from constants import ZONING_NAMES, CMAP_ZONING, CMAP_ROOMTYPE

FS = 10
AWESOME_COLORS = ["#1932E1"] # dark blue, dark green, light green, dark yellow
COLORSET = 'tab20'


def set_figure(nc, nr,
               fs=10,
               fs_title=7.5,
               fs_legend=10,
               fs_xtick=3,
               fs_ytick=3,
               fs_axes=4,
               ratio=1):
    """
    Custom figure setup function that generates a nicely looking figure outline.
    It includes "making-sense"-fontsizes across all text locations (e.g. title, axes).
    You can always change things later yourself through the outputs or plt.rc(...).
    """

    fig, axs = plt.subplots(ncols=nc, nrows=nr, figsize=(fs*nc*ratio, fs*nr))

    try:
        axs = axs.flatten()
    except:
        pass

    plt.rc("figure", titlesize=fs*fs_title)
    plt.rc("legend", fontsize=fs*fs_legend)
    plt.rc("xtick", labelsize=fs*fs_xtick)
    plt.rc("ytick", labelsize=fs*fs_ytick)
    plt.rc("axes", labelsize=fs*fs_axes, titlesize=fs*fs_title)

    return fig, axs


def plot_polygon(ax, poly, label=None, **kwargs):
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return


def plot_shapes(ax, polygons, colors, **kwargs):
    for poly, color in zip(polygons, colors):
        plot_polygon(ax, Polygon(poly), color=color, **kwargs)


def plot_graph(G, ax, c_node='black', c_edge=['white']*4, dw_edge=False, pos=None, node_size=10,
               edge_size=10):

    """
    Plots the adjacency or access graph of a floor plan's corresponding graph structure.
    """

    # position
    if pos is None:
        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node, ax=ax)

    # edges
    if dw_edge:
        epass = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'passage']
        edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'door']
        efront = [(u, v) for (u, v, d) in G.edges(data=True) if d["connectivity"] == 'entrance']
        # red full for passage, red dashed for door, yellow dashed for front
        nx.draw_networkx_edges(G, pos, edgelist=epass, edge_color=c_edge[1],
                               width=edge_size, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge[2],
                               width=edge_size, style="dashed", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=efront, edge_color=c_edge[3],
                               width=edge_size, style="-.", ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=c_edge[0],
                               width=edge_size, ax=ax)

    ax.axis('off')


def plot_floor(G, ax, node_size=50, edge_size=3):
    """Plots a floor plan's corresponding access graph. Including room shapes."""

    # Sets cmap
    attribute_names = list(G.nodes[1].keys())
    column = 'room_type' if 'room_type' in attribute_names else 'zoning_type'
    cmap = CMAP_ROOMTYPE if column == 'room_type' else CMAP_ZONING

    # Extracts node shape, color and position
    shapes = [Polygon(n) for _, n in G.nodes('geometry')]
    colors = [np.array(cmap(n)).reshape(1,4) for _, n in G.nodes(column)]
    pos = {n: np.array(G.nodes[n]['centroid']) for n in G.nodes}

    # Draw shapes
    plot_shapes(ax, shapes, colors, ec="black", lw=0, alpha=1)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='black', ax=ax)

    # Draw edges (door and passage)
    edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") if d in ["door", "passage"]]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='black',
                           width=edge_size, ax=ax)

    # Draw edges (entrance)
    edges = [(u, v) for (u, v, d) in G.edges(data="connectivity") if d == "entrance"]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red',
                           width=edge_size*2, ax=ax)
    
def plot_room_graph(apartments, graphs, viz_normals=False, viz_rooms=True, viz_walls=True, viz_openings=True, viz_wall_edges=True, viz_connection_edges=True):
    """
    Visualizes geometries, wall segments, and graph edges for multiple apartments in 2D.

    Parameters:
    apartments (list of pd.DataFrame): List of DataFrames with room geometries for each apartment.
    graphs (list of nx.Graph): List of graphs with nodes ('type', 'center', 'normal') and edges for each apartment.
    viz_normals (bool): If True, plots wall segment normals.
    viz_rooms (bool): If True, displays room polygons.
    viz_walls (bool): If True, displays wall segments.
    viz_openings (bool): If True, displays openings (doors and windows).
    viz_wall_edges (bool): If True, displays edges between wall segments.
    viz_connection_edges (bool): If True, displays edges connecting rooms via openings.

    Returns:
    plt: The matplotlib.pyplot object for further customization or display.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for apartment, G in zip(apartments, graphs):
        # Draw all room polygons
        for idx, row in apartment.iterrows():
            room_polygon = Polygon(row['geom'].exterior.coords)
            x, y = room_polygon.exterior.xy
            ax.plot(x, y, color='black', label='Room polygon' if idx == 0 else "")

        # Draw the room centroids
        if viz_rooms:
            room_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'room']
            for idx, room_node in enumerate(room_nodes):
                room_data = G.nodes[room_node]
                ax.scatter(room_data['center'][0], room_data['center'][1], color='blue', s=100, label='Room centroid' if idx == 0 else "")

        # Draw the wall segments and their normals
        if viz_walls:
            wall_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'ws']
            for idx, wn in enumerate(wall_nodes):
                wall_data = G.nodes[wn]
                ax.scatter(wall_data['center'][0], wall_data['center'][1], color='red', s=50, label='Wall segment' if idx == 0 else "")
                if viz_normals:
                    ax.arrow(wall_data['center'][0], wall_data['center'][1],
                             wall_data['normal'][0], wall_data['normal'][1],
                             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal' if idx == 0 and not viz_normals else "")      

        # Draw the openings and their normals
        if viz_openings:
            opening_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'door' or d['type'] == 'window']
            for idx, on in enumerate(opening_nodes):
                opening_data = G.nodes[on]
                ax.scatter(opening_data['center'][0], opening_data['center'][1], color='orange', s=50, label='Opening' if idx == 0 else "")
                if viz_normals:
                    ax.arrow(opening_data['center'][0], opening_data['center'][1],
                             opening_data['normal'][0], opening_data['normal'][1],
                             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal' if idx == 0 and not viz_normals else "") 

        # Draw edges
        for idx, edge in enumerate(G.edges(data=True)):
            start_node = G.nodes[edge[0]]
            end_node = G.nodes[edge[1]]
            
            # Filter edges based on visualization preferences
            if not viz_wall_edges and ('belong' in edge[2]['type'] or 'same' in edge[2]['type']):
                continue
            if not viz_connection_edges and ('connected_by' in edge[2]['type']):
                continue
            if not viz_rooms and ('room' in start_node['type'] or 'room' in end_node['type']):
                continue
            if not viz_walls and ('ws' in start_node['type'] or 'ws' in end_node['type']):
                continue
            if not viz_openings and ('door' in start_node['type'] or 'door' in end_node['type'] or 
                                      'window' in start_node['type'] or 'window' in end_node['type']):
                continue
            
            edge_color = 'purple' if 'opening' in edge[2] else 'gray'  # Use purple for edges with 'opening'
            ax.plot([start_node['center'][0], end_node['center'][0]],
                    [start_node['center'][1], end_node['center'][1]],
                    color=edge_color, linestyle='--', label='Edge' if idx == 0 else "")

    plt.title("Apartments with geometries, wall segments, and edges in 2D")
    plt.legend()
    return plt

def plot_DF_geometries(floor_DF):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot by entity_type
    floor_DF.plot(ax=axs[0], column='entity_type', legend=True)
    axs[0].set_title('Rooms, Separators, and Openings by Type')

    # Plot by entity_subtype
    # Generate a unique color for each subtype
    unique_subtypes = floor_DF['entity_subtype'].unique()
    subtype_colors = {subtype: plt.cm.tab20(i / len(unique_subtypes)) for i, subtype in enumerate(unique_subtypes)}

    # Plot by entity_subtype with custom colors
    floor_DF.plot(ax=axs[1], column='entity_subtype', legend=True, cmap=mcolors.ListedColormap([subtype_colors[subtype] for subtype in unique_subtypes]))
    axs[1].set_title('Rooms, Separators, and Openings by Subtype')

    plt.tight_layout()
    plt.show()