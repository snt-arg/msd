from itertools import combinations
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point
import torch
import networkx as nx


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


def get_geometries_from_id(df, floor_id, column='zoning'):

    """
    Extracting geometry information from particular floor ID.
    """

    df_floor = df[(df.floor_id == floor_id)].reset_index(drop=True)
    df_floor.geom = df_floor.geom.apply(wkt.loads)

    geoms, geoms_type = zip(*df_floor[["geom", column]].values)

    return geoms, geoms_type

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