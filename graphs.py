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


def extract_access_graph(geoms, geoms_type, classes, id):
    """Extracts the access graph from a set of geometries."""

    # Sets the mapping
    mapping = {cat: index for index, cat in enumerate(classes)}

    # Initializes and separate areas (and types), doors, and entrance doors
    areas, areas_type, doors, entrance_doors = [], [], [], []

    # Makes sure to only have {Zone1, Zone2, Zone3, Zone4} in areas
    end = 4 if 'Zone1' in classes else 9
    for geom, geom_type in zip(geoms, geoms_type):
        if geom_type == 'Door':
            doors.append(geom)
        elif geom_type == 'Entrance Door':
            entrance_doors.append(geom)
        elif geom_type in classes[:end]:
            areas.append(geom)
            areas_type.append(geom_type)
        else: continue  # walls are omitted

    # Accumulate nodes
    area_nodes = {}
    for key, (area, area_type) in enumerate(zip(areas, areas_type)):

        # Zoning (input) graph node attributes
        if 'Zone1' in classes:
            area_nodes[key] = {
                'zoning_type': mapping[area_type]
            }
        # Full (output) graph attributes
        else:
            area_nodes[key] = {
                'geometry': polygon_to_list(area),
                'room_type': mapping[area_type],
                'centroid': torch.tensor(np.array([area.centroid.x, area.centroid.y]))
            }

    # Accumulate edges
    edges = []
    for (i, v1), (j, v2) in combinations(enumerate(areas), 2):

        # Option 1: PASSAGE (direct access := no wall in between)
        if v1.distance(v2) < 0.04:
            edges.append([i, j, {'connectivity': 'passage'}])

        # Option 2: DOOR
        else:
            for door in doors:
                if door.distance(v1) < 0.05 and door.distance(v2) < 0.05:
                    # Adds the geometry of the door as well (slightly different from paper)
                    edges.append([i, j, {'connectivity': 'door'}])  #, 'door_geometry': polygon_to_list(door)}])
                else: continue

        # Option 3: FRONT DOOR
        for entrance_door in entrance_doors:
            if entrance_door.distance(v1) < 0.05 and entrance_door.distance(v2) < 0.05:
                # Adds the geometry of the door as well (slightly different from paper)
                edges.append([i, j, {'connectivity': 'entrance'}])  #, 'door_geometry': polygon_to_list(entrance_door)}])
            else: continue

    # Defines the graph
    G = nx.Graph()
    G.graph["ID"] = id  # Give the floor ID as graph attribute
    G.add_nodes_from([(u, v) for u, v in area_nodes.items()])
    G.add_edges_from(edges)

    return G


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
    room_id = f"{floor_id}_{apartment_id}_{category_letter}{key}_centroid"
    #TODO: maybe its better to change change z and set the norm of z norm = 1 or something else
    G.add_node(room_id, polygon = polygon_to_list(polygon), center=[polygon.centroid.x, polygon.centroid.y, z],
                normal=[0, 0, 0], type='room', category= category, category_letter = category_letter)

    node_ids = []

    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])

        # Midpoint of the segment
        midpoint = (p1 + p2) / 2

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

        node_id = f"{floor_id}_{apartment_id}_{category_letter}{key}_ws_{i}"
        node_ids.append(node_id)

        G.add_node(node_id, center=midpoint_3d.tolist(), normal=normal_3d.tolist(), type='ws')

    # Add edges between consecutive segments and the room centroid
    for i in range(len(node_ids) - 1):
        G.add_edge(node_ids[i], node_ids[i + 1], type='ws_same_room')
        G.add_edge(node_ids[i], room_id, type='ws_belong_room')

    # Close the loop by connecting last to first
    G.add_edge(node_ids[-1], node_ids[0], type='ws_same_room')
    G.add_edge(node_ids[-1], room_id, type='ws_belong_room')

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
    room_id = f"{floor_id}_{apartment_id}_{room_entity_subtype}{key}_centroid"

    def process_openings(openings, opening_indexes, openings_z, opening_type):
        for idx in opening_indexes:
            polygon = openings[idx]
            coords = list(polygon.exterior.coords)
            z = openings_z[idx]

            opening_id = f"{floor_id}_{apartment_id}_{opening_type}_{idx}"
            node_ids = []

            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])

                # Midpoint of the segment
                midpoint = (p1 + p2) / 2

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

                    G.add_node(node_id, center=midpoint_3d.tolist(), normal=normal_3d, type=opening_type)

            # Add edges to the room centroid
            for node_id in node_ids:
                G.add_edge(node_id, room_id, type=f'{opening_type}_belong_room')
    
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

def extract_a_graph(geoms, cats, names, apartment_id, floor_id):
    """Extracts the access graph from a set of geometries."""

    # Sets the mapping
    mapping_names = {cat: i for i, cat in enumerate(names)}

    # Initializes empty lists for rooms and their categories, doors, and walls
    rooms, room_cats, doors, entrances, walls, windows = [], [], [], [], [], []

    # Loops through the geometries and corresponding categories
    for geom, cat in zip(geoms, cats):
        if cat ==  'Door':  # Doors
            doors.append(geom)
        elif cat ==  'Entrance Door':  # Entrances
            doors.append(geom)
            entrances.append(geom)
        elif cat in names[:9]:  # Rooms
            rooms.append(geom)
            room_cats.append(cat)
        elif cat == 'Structure':  # Walls and columns
            walls.append(geom)
        elif cat == 'Window': # Windows
            windows.append(geom)
        else: continue

    # Accumulation of NODES (i.e., the rooms)
    anodes = {}
    aedges = []

    number_of_rooms = len(rooms)
    walls_start_index = number_of_rooms
    
    for key, (room, cat) in enumerate(zip(rooms, room_cats)):
        
        points = polygon_to_list(room)
        centroid = np.array([room.centroid.x, room.centroid.y])
        
        #add categoy, type and centroid to anode
        anodes[key] = {
            'polygon': points,
            'category': mapping_names[cat],
            'type': 'room',
            'centroid': torch.tensor(centroid),
            'normal': np.array([0, 0])
        }

        
        # for each wall calculate the mid point and the normal vector
        for j in range(len(points)-1):
        
            # mid point
            x = (points[j][0]+points[j+1][0])/2
            y = (points[j][1]+points[j+1][1])/2
            # normal vector
            normalx = x - centroid[0]
            normaly = y - centroid[1]

            # normalize the normal vector 
            if(abs(normalx) > abs(normaly)):
                normalx = 1
                normaly = 0
            else:
                normalx = 0
                normaly = 1
                
            # add wall to anodes
            anodes[walls_start_index + j] = {
                'polygon': [],
                'category': mapping_names[cat],
                'type': 'wall',
                'centroid': torch.tensor(np.array([x, y])),
                'normal': np.array([normalx, normaly])
            }

        walls_end_index = walls_start_index + len(points)-1
    
        # for each wall nodes create an edge with the room node and with the next wall node
        for i in range(walls_start_index, walls_end_index):
            aedges.append([i, key, {'type' : 'ws_belongs_room'}])
            aedges.append([i, i+1, {'type' : 'ws_same_room'}])

        walls_start_index = walls_end_index

    # Accumulation of EDGES (i.e., room to room connectivity)
    for (i, v1), (j, v2) in combinations(enumerate(rooms), 2):

        # (Option 1) Passage (i.e., direct access := no wall in between)
        if v1.distance(v2) < 0.04:
            aedges.append([i, j, {'polygon': None, 'connectivity': 'passage'}])
        
        # TODO connect rooms through doors and windows nodes
        # (Option 2) Door (i.e., door in between two rooms)
        else:
            edge = False
            for door in doors + entrances:
                door_rotated = rotate_rectangle(door, scale_factor=1)
                if door_rotated.intersection(v1) and door_rotated.intersection(v2):
                    # Adds the geometry of the door as well (slightly different from paper)
                    edge = True
                    aedges.append([i, j, {'polygon': polygon_to_list(door), 'connectivity': 'door'}])
                else: continue

            # (Option 2B) By window (i.e., window between balcony and other room)
            # Sometimes, balconies seem disconnected from the apartment (fully).
            # This is likely not the case. So, if a balcony connects with one of the other rooms
            # through a window it is fine as well.
            if not edge and (room_cats[i] == "Balcony" or room_cats[j] == "Balcony"):
                # Check connection based on window overlap
                for window in windows:
                    window_rotated = rotate_rectangle(window)
                    if window_rotated.intersection(v1) and window_rotated.intersection(v2):
                        aedges.append([i, j, {'polygon': polygon_to_list(window), 'connectivity': 'door'}])
                    else: continue

    # Get tightest boundary of the apartment
    # (1) Unite all these wall geometries
    # (2) Find the polygon within the union that is largest in terms of area (using np.argsort)
    #   and choose the largest (which is by default put on the end of the sort)
    structure = unary_union(walls)  # (1)
    if structure.geom_type == "MultiPolygon":
        boundary = structure.geoms[np.argsort([geom.area for geom in structure.geoms])[-1]]  # (2)
    elif structure.geom_type == "Polygon":
        boundary = deepcopy(structure)
    else:
        raise NotImplementedError(f"Not implemented for {structure.geom_type}.")

    #add anodes and aedges to the A graph
    AG = nx.Graph()
    # Graph attributes / features
    AG.graph["Floor ID"] = floor_id  # Floor ID
    AG.graph["Apt ID"] = apartment_id  # Apartment ID (i.e., name)
    AG.graph["Structure"] = walls  # Walls and columns
    AG.graph["Windows"] = windows  # Windows
    AG.graph["Entrances"] = entrances  # Entrances (doors)
    # Node attributes / features
    AG.add_nodes_from([(u, v) for u, v in anodes.items()])
    # Edge attributes / features
    AG.add_edges_from(aedges)

    return AG