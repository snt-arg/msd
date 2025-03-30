from itertools import combinations
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point
import torch
import networkx as nx
from copy import deepcopy
from shapely.affinity import rotate, scale
from shapely.ops import unary_union


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))


def polygon_to_array(polygon: Polygon) -> np.array:
    """Converts a polygon into a numpy array."""
    return np.array(polygon_to_list(polygon))


# def extract_access_graph(geoms, geoms_type, classes, id):
#     """Extracts the access graph from a set of geometries."""

#     # Sets the mapping
#     mapping = {cat: index for index, cat in enumerate(classes)}

#     # Initializes and separate areas (and types), doors, and entrance doors
#     areas, areas_type, doors, entrance_doors = [], [], [], []

#     # Makes sure to only have {Zone1, Zone2, Zone3, Zone4} in areas
#     end = 4 if 'Zone1' in classes else 9
#     for geom, geom_type in zip(geoms, geoms_type):
#         if geom_type == 'Door':
#             doors.append(geom)
#         elif geom_type == 'Entrance Door':
#             entrance_doors.append(geom)
#         elif geom_type in classes[:end]:
#             areas.append(geom)
#             areas_type.append(geom_type)
#         else: continue  # walls are omitted

#     # Accumulate nodes
#     area_nodes = {}
#     for key, (area, area_type) in enumerate(zip(areas, areas_type)):

#         # Zoning (input) graph node attributes
#         if 'Zone1' in classes:
#             area_nodes[key] = {
#                 'zoning_type': mapping[area_type]
#             }
#         # Full (output) graph attributes
#         else:
#             area_nodes[key] = {
#                 'geometry': polygon_to_list(area),
#                 'room_type': mapping[area_type],
#                 'centroid': torch.tensor(np.array([area.centroid.x, area.centroid.y]))
#             }

#     # Accumulate edges
#     edges = []
#     for (i, v1), (j, v2) in combinations(enumerate(areas), 2):

#         # Option 1: PASSAGE (direct access := no wall in between)
#         if v1.distance(v2) < 0.04:
#             edges.append([i, j, {'connectivity': 'passage'}])

#         # Option 2: DOOR
#         else:
#             for door in doors:
#                 if door.distance(v1) < 0.05 and door.distance(v2) < 0.05:
#                     # Adds the geometry of the door as well (slightly different from paper)
#                     edges.append([i, j, {'connectivity': 'door'}])  #, 'door_geometry': polygon_to_list(door)}])
#                 else: continue

#         # Option 3: FRONT DOOR
#         for entrance_door in entrance_doors:
#             if entrance_door.distance(v1) < 0.05 and entrance_door.distance(v2) < 0.05:
#                 # Adds the geometry of the door as well (slightly different from paper)
#                 edges.append([i, j, {'connectivity': 'entrance'}])  #, 'door_geometry': polygon_to_list(entrance_door)}])
#             else: continue

#     # Defines the graph
#     G = nx.Graph()
#     G.graph["ID"] = id  # Give the floor ID as graph attribute
#     G.add_nodes_from([(u, v) for u, v in area_nodes.items()])
#     G.add_edges_from(edges)

#     return G


def get_geometries_from_id(df, floor_id, apartment_id=None, column='roomtype'):
    """Function that extracts all geometries and associated categories from a
    particular floor and apartment ID. Outputs a list of geometries (1) and
    a corresponding list of categories (2)."""

    # Samples dataframe based on floor and, if asked for, apartment ID
    df_floor = df[df.floor_id == floor_id].reset_index(drop=True)
    if  apartment_id is not None:
        df_floor = df_floor[df_floor.apartment_id == apartment_id].reset_index(drop=True)
    else:
        pass
    df_floor.geom = df_floor.geom.apply(wkt.loads)

    # Get geometries and associated categories out
    geoms, cats = zip(*df_floor[["geom", column]].values)

    return geoms, cats

# For each segment of the room geometry, find the midpoint and inward-pointing normal
def get_segment_normals_toward_inside(geom, epsilon=1e-3):
    """
    Input:
        geom: A GeoDataFrame row containing a polygon geometry and attributes like 'elevation' and 'height'.
        epsilon: A small value to test the inward direction of the normal vector (default is 1e-3).
    Output:
        A list of dictionaries, where each dictionary contains:
            - 'center': The 3D coordinates of the midpoint of a segment.
            - 'normal': The 3D inward-pointing normal vector of the segment.
    Computes the midpoints and inward-facing normal vectors of each segment of the polygon's exterior boundary.
    The normals are adjusted to point towards the interior of the polygon.
    """

    polygon = geom.geometry

    coords = list(polygon.exterior.coords)
    results = []

    # get elevation and height of the room
    elevation = geom['elevation']
    height = geom['height']
    z = (elevation + height) / 2

    for i in range(len(coords) - 1):  # -1 because the last point repeats the first
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])

        # Midpoint of the segment
        midpoint = (p1 + p2) / 2

        # Edge vector
        edge_vec = p2 - p1

        # Perpendicular normal (rotated 90°)
        normal = np.array([-edge_vec[1], edge_vec[0]])

        # Normalize the normal vector
        normal /= np.linalg.norm(normal)

        # Test point slightly offset in the direction of the normal
        p_test = midpoint + epsilon * normal

        # If the point is NOT inside the polygon, invert the normal direction
        if not polygon.contains(Point(p_test)):
            normal = -normal

        #convert to 3D
        midpoint = np.append(midpoint, z)
        normal = np.append(normal, 0)

        results.append({
            'center': midpoint,
            'normal': normal
        })

    return results

def add_room_geometries(geom_dict, floor_id, apartment_id, key, G, epsilon=1e-3):
    """
    Creates a directed graph (DiGraph) from the segments of the outer contour of a polygon.
    Each node represents the midpoint of a segment and contains the inward normal in 3D.
    A node is also created for the room centroid with its elevation.
    Edges connect consecutive wall segment nodes and link each segment to the room centroid.

    Input:
        geom_dict: A dictionary containing 'geom' (Polygon), 'entity_subtype' (str), and 'z' (float).
        G: A directed graph (DiGraph) to which the nodes and edges will be added.
        epsilon: Value to verify the direction of the normal (default 1e-3).
    """

    polygon = geom_dict['geom']
    coords = list(polygon.exterior.coords)
    z = geom_dict['z']
    category = geom_dict['category']
    category_letter = geom_dict['entity_subtype']


    # Create a node for the room with the room ID and the centroid
    room_id = f"{floor_id}_{apartment_id}_{category_letter}_{key}_centroid"
    # Set norm z = 1 to force the difference from other nodes 
    G.add_node(room_id, polygon = polygon_to_list(polygon), center=[polygon.centroid.x, polygon.centroid.y, z],
                normal=[0, 0, 1], type='room', category=category, category_letter = category_letter)

    node_ids = []

    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])

        # Midpoint of the segment
        midpoint = (p1 + p2) / 2

        # calculate euclidean distance
        width = np.linalg.norm(p2 - p1)

        # Edge vector
        edge_vec = p2 - p1

        # Perpendicular normal
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)

        # Test inward direction
        p_test = midpoint + epsilon * normal
        if not polygon.contains(Point(p_test)):
            normal = -normal

        # Convert to 3D
        midpoint_3d = np.append(midpoint, z)
        normal_3d = np.append(normal, 0)

        node_id = f"{floor_id}_{apartment_id}_{category_letter}_{key}_ws_{i}"
        node_ids.append(node_id)
        # Create a polygon with p1 and p2

        G.add_node(node_id, geom=[p1,p2], polygon = polygon_to_list(Polygon([p1, p2, p2 + [0, 0.01], p1 + [0, 0.01]])),
                    center=midpoint_3d.tolist(), normal=normal_3d.tolist(), width=width, type='ws', category=9)

    # Add edges between consecutive segments and the room centroid
    for i in range(len(node_ids) - 1):
        G.add_edge(node_ids[i], node_ids[i + 1], type='ws_same_room')
        G.add_edge(node_ids[i], room_id, type='ws_belongs_room')

    # Close the loop by connecting last to first
    G.add_edge(node_ids[-1], node_ids[0], type='ws_same_room')
    G.add_edge(node_ids[-1], room_id, type='ws_belongs_room')

    return

def add_opening_geometry(doors, windows, door_indexes, window_indexes, doors_z, windows_z, room_dict, floor_id, apartment_id, key, G, epsilon=1e-3):
    """
    Adds nodes and edges for openings (e.g., doors, windows) to a directed graph (DiGraph).
    Each node represents the midpoint of a segment of the opening and contains the outward normal in 3D,
    constrained to remain inside the containing room.

    Input:
        doors: A list of Polygon geometries representing the doors.
        windows: A list of Polygon geometries representing the windows.
        door_indexes: A list of indexes to access the doors of interest.
        window_indexes: A list of indexes to access the windows of interest.
        doors_z: A list of z-values (elevation) corresponding to the doors.
        windows_z: A list of z-values (elevation) corresponding to the windows.
        room_dict: A dictionary containing 'geom' (Polygon), 'entity_subtype' (str), 'category' (int), and 'z' (float).
        floor_id: The floor ID.
        apartment_id: The apartment ID.
        G: A directed graph (DiGraph) to which the nodes and edges will be added.
        epsilon: Value to verify the direction of the normal (default 1e-3).
    """

    room_polygon = room_dict['geom']  # Extract the room geometry
    room_entity_subtype = room_dict['entity_subtype']
    room_id = f"{floor_id}_{apartment_id}_{room_entity_subtype}_{key}_centroid"

    def process_openings(openings, opening_indexes, openings_z, opening_type):
        for idx in opening_indexes:
            polygon = openings[idx]
            coords = list(polygon.exterior.coords)
            z = openings_z[idx]

            opening_id = f"{floor_id}_{apartment_id}_{opening_type}_{idx}"
            
            # Set norm z = 1 to force the difference from other ws nodes 
            G.add_node(opening_id, polygon = polygon_to_list(polygon), center=[polygon.centroid.x, polygon.centroid.y, z],
                        normal=[0, 0, 1], type=opening_type)

            node_ids = []

            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])

                # Midpoint of the segment
                midpoint = (p1 + p2) / 2

                # calculate euclidean distance
                width = np.linalg.norm(p2 - p1)

                # Edge vector
                edge_vec = p2 - p1

                # Perpendicular normal
                normal = np.array([-edge_vec[1], edge_vec[0]])
                normal /= np.linalg.norm(normal)

                # Test outward direction
                test_point = midpoint + epsilon * normal
                if polygon.contains(Point(test_point)):
                    normal = -normal  # flip the direction if it's pointing inward or outside the room

                # Check if the normal is inside the room
                if not room_polygon.contains(Point(midpoint + epsilon * normal)):
                    normal = [0, 0]  # set to zero if it's not inside the room

                # Add the node only if the normal is valid
                if np.linalg.norm(normal) > 0:
                    midpoint_3d = np.append(midpoint, z)
                    normal_3d = np.append(normal, 0)

                    node_id = f"{opening_id}_{i}"
                    node_ids.append(node_id)

                    G.add_node(node_id, geom=[p1,p2], polygon = polygon_to_list(Polygon([p1, p2, p2 + [0, 0.01], p1 + [0, 0.01]])),
                                center=midpoint_3d.tolist(), normal=normal_3d, width=width,  type=f"{opening_type}_ws", category=9)

            # Add edges to the room centroid
            for node_id in node_ids:
                G.add_edge(node_id, room_id, type=f"{opening_type}_belongs_room")
                G.add_edge(node_id, opening_id, type=f"ws_belongs_{opening_type}")
    
    # Process doors if not empty
    if doors:
        process_openings(doors, door_indexes, doors_z, 'door')

    # Process windows if not empty
    if windows:
        process_openings(windows, window_indexes, windows_z, 'window')

    return

def get_segment_normals_outward_inside_room(geom, room_geom, epsilon=1e-3):
    """
    For each edge of the geometry,
    compute the midpoint and a normal vector pointing outward from the object
    only if it remain inside the containing room.
    
    Input:
        geom: A GeoDataFrame row containing a polygon geometry and attributes like 'elevation' and 'height'.
        room_geom: The geometry of the containing room.
        epsilon: A small value to test the outward direction of the normal vector (default is 1e-3).

    Output:
        A dictionary containing:
            - 'midpoint': The 3D coordinates of the midpoint of a segment.
            - 'normal': The 3D outward-pointing normal vector of the segment, constrained to remain inside the room.
    """

    polygon = geom.geometry
    coords = list(polygon.exterior.coords)
    results = {}
    
    # get elevation and height of the geom
    elevation = geom['elevation']
    height = geom['height']
    z = (elevation + height) / 2

    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])

        midpoint = (p1 + p2) / 2
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)

        test_point = midpoint + epsilon * normal

        # Check if the normal is outward from the object
        if polygon.contains(Point(test_point)):
            normal = -normal  # flip the direction if it's pointing inward or outside the room

        # Check if the normal is inside the room
        if not room_geom.contains(Point(midpoint + epsilon * normal)):
            normal = [0, 0]  # set to zero if it's not inside the room
        
        # Append the result only if the normal is not zero
        if np.linalg.norm(normal) > 0: 
            #convert to 3D
            midpoint = np.append(midpoint, z)
            normal = np.append(normal, 0)

            results = {
                'midpoint': midpoint,
                'normal': normal
            }

    return results

def rotate_rectangle(rect: Polygon, scale_factor=0.5, angle=90):
    # Compute centroid (center point)
    centroid = rect.centroid

    # Scale the rectangle (relative to the centroid)
    scaled_rect = scale(rect, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    # Rotate the scaled rectangle around its center
    rotated_rect = rotate(scaled_rect, angle, origin=centroid)

    return rotated_rect

def connect_area_by_openings(graph,apartment_id):
    for opening in ["door", "window"]:
        # Extract unique opening IDs, ei. 'window_1', 'window_0'
        ids = list(set(v.split("_")[2] + "_" + v.split("_")[3] 
                       for u, v in graph.edges() if opening in u or opening in v))
        print(f"Opening IDs: {ids}")

        for id in ids:
        
            # Find areas connected by the current opening
            connected_areas = [u.split("_")[2] + "_" + u.split("_")[3] 
                               for u, v in graph.edges() if (id in u or id in v) and not (id in v and id in u)]

            print(f"Connected areas for opening {id}: {connected_areas}")

            # Create edges between all pairs of connected areas
            for i, j in combinations(connected_areas, 2):
                room1 = apartment_id + "_" + i + "_centroid"
                room2 = apartment_id + "_" + j + "_centroid"
                graph.add_edge(room1, room2, type=f"connected_by_{opening}", opening=id)



def extract_access_graph(geoms, cats, elevations, heights, names, apartment_id, floor_id):
    """Extracts the access graph from a set of apartment."""

    # Defines the graph
    G = nx.Graph()

    # Sets the mapping
    mapping_names = {cat: i for i, cat in enumerate(names)}

    # Initializes empty lists for rooms and their categories, doors, and walls
    rooms, room_cats, doors, entrances, walls, windows = [], [], [], [], [], []
    room_z, doors_z, entrances_z, walls_z, windows_z = [], [], [], [], []

    # Loops through the geometries and corresponding categories
    for geom, cat, height in zip(geoms, cats, heights):

        # Add z-coordinate to the geometry
        z_coord = height / 2

        if cat == 'Door':  # Doors
            doors.append(geom)
            doors_z.append(z_coord)
        elif cat == 'Entrance Door':  # Entrances
            doors.append(geom)
            doors_z.append(z_coord)
            entrances.append(geom)
            entrances_z.append(z_coord)
        elif cat in names[:9]:  # Rooms
            rooms.append(geom)
            room_z.append(z_coord)
            room_cats.append(cat)
        elif cat == 'Structure':  # Walls and columns
            walls.append(geom)
            walls_z.append(z_coord)
        elif cat == 'Window':  # Windows
            windows.append(geom)
            windows_z.append(z_coord)
        else:
            continue


    for key, (room, cat, z) in enumerate(zip(rooms, room_cats, room_z)):
        # for each room in the apartment

        # create dict for room, cat, and z
        room_dict = {
            'geom': room,
            'entity_subtype': cat,
            'category': mapping_names[cat],
            'z': z
        }

        # search for intersections with openings
        wall_indexes, door_indexes, window_indexes = ut.find_intersections(room, walls, doors, windows)

        gr.add_room_geometries(room_dict, floor_id, apartment_id, key, G)

        gr.add_opening_geometry(doors, windows, door_indexes, window_indexes, doors_z, windows_z, room_dict, floor_id, apartment_id, key, G)
            
    gr.connect_area_by_openings(G, str(str(floor_id) + "_" + str(apartment_id)))

    # Graph attributes / features
    G.graph["Floor ID"] = floor_id  # Floor ID
    G.graph["Apt ID"] = apartment_id  # Apartment ID (i.e., name)
    G.graph["Structure"] = walls  # Walls and columns
    G.graph["Windows"] = windows  # Windows
    G.graph["Entrances"] = entrances  # Entrances (doors)
    
    return G