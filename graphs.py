from itertools import combinations
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point
import torch
import networkx as nx
from copy import deepcopy
from shapely.affinity import rotate, scale
from shapely.ops import unary_union
import utils as ut


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))


def polygon_to_array(polygon: Polygon) -> np.array:
    """Converts a polygon into a numpy array."""
    return np.array(polygon_to_list(polygon))

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

        if width == 0:
            continue

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

def add_other_geometry(doors, windows, walls, door_indexes, window_indexes, wall_indexes, doors_z, windows_z, walls_z, room_dict, floor_id, apartment_id, key, G, epsilon=1e-3):
    """
    Adds nodes and edges for entities (e.g., doors, windows, walls) to a directed graph (DiGraph).
    Each node represents the midpoint of a segment of the entity and contains the outward normal in 3D,
    constrained to remain inside the containing room.

    Input:
        doors: A list of Polygon geometries representing the doors.
        windows: A list of Polygon geometries representing the windows.
        walls: A list of Polygon geometries representing the walls.
        door_indexes: A list of indexes to access the doors of interest.
        window_indexes: A list of indexes to access the windows of interest.
        wall_indexes: A list of indexes to access the walls of interest.
        doors_z: A list of z-values (elevation) corresponding to the doors.
        windows_z: A list of z-values (elevation) corresponding to the windows.
        walls_z: A list of z-values (elevation) corresponding to the walls.
        room_dict: A dictionary containing 'geom' (Polygon), 'entity_subtype' (str), 'category' (int), and 'z' (float).
        floor_id: The floor ID.
        apartment_id: The apartment ID.
        G: A directed graph (DiGraph) to which the nodes and edges will be added.
        epsilon: Value to verify the direction of the normal (default 1e-3).
    """

    room_polygon = room_dict['geom']  # Extract the room geometry
    room_entity_subtype = room_dict['entity_subtype']
    room_id = f"{floor_id}_{apartment_id}_{room_entity_subtype}_{key}_centroid"

    def process_entity(entities, entity_indexes, entities_z, entity_type, eps):
        for idx in entity_indexes:
            polygon = entities[idx]
            coords = list(polygon.exterior.coords)
            z = entities_z[idx]

            entity_id = f"{floor_id}_{apartment_id}_{entity_type}_{idx}"

            # Set norm z = 1 to force the difference from other ws nodes 
            G.add_node(entity_id, polygon = polygon_to_list(polygon), center=[polygon.centroid.x, polygon.centroid.y, z],
                        normal=[0, 0, 1], type=entity_type)

            node_ids = []

            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])

                # Midpoint of the segment
                midpoint = (p1 + p2) / 2

                # calculate euclidean distance
                width = np.linalg.norm(p2 - p1)

                if width == 0:
                    continue

                # Edge vector
                edge_vec = p2 - p1

                # Perpendicular normal
                normal = np.array([-edge_vec[1], edge_vec[0]])
                normal /= np.linalg.norm(normal)

                # Test outward direction
                test_point = midpoint + eps * normal
                if polygon.contains(Point(test_point)):
                    normal = -normal  # flip the direction if it's pointing inward or outside the room

                # Check if the normal is inside the room
                if not room_polygon.contains(Point(midpoint + eps * normal)):
                    normal = [0, 0]  # set to zero if it's not inside the room

                # Add the node only if the normal is valid
                if np.linalg.norm(normal) > 0:
                    midpoint_3d = np.append(midpoint, z)
                    normal_3d = np.append(normal, 0)

                    node_id = f"{entity_id}_{i}"
                    node_ids.append(node_id)

                    G.add_node(node_id, geom=[p1,p2], polygon = polygon_to_list(Polygon([p1, p2, p2 + [0, 0.01], p1 + [0, 0.01]])),
                                center=midpoint_3d.tolist(), normal=normal_3d, width=width,  type=f"{entity_type}_ws", category=9)

            # Add edges to the room centroid
            for node_id in node_ids:
                G.add_edge(node_id, room_id, type=f"{entity_type}_belongs_room")
                G.add_edge(node_id, entity_id, type=f"ws_belongs_{entity_type}")
    
    # Process doors if not empty
    if doors:
        process_entity(doors, door_indexes, doors_z, 'door', epsilon)

    # Process windows if not empty
    if windows:
        process_entity(windows, window_indexes, windows_z, 'window', epsilon)

    # Process walls if not empty
    if walls:
        process_entity(walls, wall_indexes, walls_z, 'wall', 1e-2)

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

#Not working good
def connect_area_by_openings(graph,apartment_id):
    for opening in ["door"]:
        # Extract unique opening IDs, ei. 'door_1', 'door_0'
        ids = list(set(v.split("_")[2] + "_" + v.split("_")[3] 
                       for u, v in graph.edges() if opening in u or opening in v))
        print(f"Opening IDs: {ids}")

        for id in ids:
        
            # Find areas connected by the current opening
            connected_areas = [u.split("_")[2] + "_" + u.split("_")[3] 
                               for u, v in graph.edges() if (id in u or id in v) and not (id in v and id in u)]
            # Remove duplicates
            connected_areas = list(set(connected_areas))
            
            print(f"Connected areas for opening {id}:{connected_areas}")

            # assume an opening can connect only 2 areas
            if len(connected_areas) == 2:
                room1 = apartment_id + "_" + connected_areas[0] + "_centroid"
                room2 = apartment_id + "_" + connected_areas[1] + "_centroid"
                graph.add_edge(room1, room2, type=f"connected_by_{opening}", opening=id)
            elif len(connected_areas) < 2:
                continue
            else:
                print(f"Warning: More than 2 areas connected by opening {id}: {connected_areas}")
                # Create edges between all pairs of connected areas
                for i, j in combinations(connected_areas, 2):
                    room1 = apartment_id + "_" + i + "_centroid"
                    room2 = apartment_id + "_" + j + "_centroid"
                    graph.add_edge(room1, room2, type=f"connected_by_{opening}", opening=id)

def connect_rooms_by_proximity(G, doors, entrances, windows):
    """
    Connects room nodes in the graph based on proximity or openings (doors/windows).

    Input:
        G: The graph containing room nodes.
        apartment_id: The ID of the apartment.
        doors: List of door geometries.
        entrances: List of entrance geometries.
        windows: List of window geometries.
    """
    # Extract all room nodes
    room_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'room']

    # Iterate over all pairs of room nodes
    for i, j in combinations(room_nodes, 2):
        v1 = Polygon(G.nodes[i]['polygon'])
        v2 = Polygon(G.nodes[j]['polygon'])

        # (Option 1) Passage (direct access, no wall in between)
        if v1.distance(v2) < 0.04:
            G.add_edge(i, j, type=f"connected_by_passage", opening='passage')

        # (Option 2) Door (door in between two rooms)
        else:
            edge = False
            for door in doors + entrances:
                door_rotated = rotate_rectangle(door, scale_factor=1)
                if door_rotated.intersection(v1) and door_rotated.intersection(v2):
                    edge = True
                    G.add_edge(i, j, type=f"connected_by_door", opening='door')
                else:
                    continue

            # (Option 2B) By window (window between balcony and other room)
            if not edge and (G.nodes[i].get('category_letter') == "Balcony" or G.nodes[j].get('category_letter') == "Balcony"):
                for window in windows:
                    window_rotated = rotate_rectangle(window)
                    if window_rotated.intersection(v1) and window_rotated.intersection(v2):
                        G.add_edge(i, j, type=f"connected_by_window", opening='window')
                    else:
                        continue
