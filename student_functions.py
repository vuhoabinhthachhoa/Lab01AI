import numpy as np
from queue import Queue, PriorityQueue
import math

def BFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO: 

    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    queue = Queue()  # Initialize the queue for BFS
    queue.put(start)  # Enqueue the start node
    visited[start] = None  # Mark the start node as visited with no predecessor

    while not queue.empty():  # While the queue is not empty
        node = queue.get()  # Dequeue a node
        if node == end:  # If the end node is reached, break
            break
        for neighbor, connected in enumerate(matrix[node]):  # Iterate through every neighbors nodes
            if connected and neighbor not in visited:  # If connected and not visited
                queue.put(neighbor)  # Enqueue the neighbor
                visited[neighbor] = node  # Mark the neighbor as visited with predecessor

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path

def DFS(matrix, start, end):
    """
    DFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # TODO: 

    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    stack = [start]  # Initialize the stack for DFS
    visited[start] = None  # Mark the start node as visited with no predecessor

    while stack:  # While the stack is not empty
        node = stack.pop()  # Pop a node from the stack
        if node == end:  # If the end node is reached, break
            break
        for neighbor, connected in enumerate(matrix[node]):  # Iterate over neighbors
            if connected and neighbor not in visited:  # If connected and not visited
                stack.append(neighbor)  # Push the neighbor onto the stack
                visited[neighbor] = node  # Mark the neighbor as visited with predecessor

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path

def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
     Parameters:visited
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  

    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    pq = PriorityQueue()  # Initialize the priority queue for UCS
    pq.put((0, start))  # Enqueue the start node with cost 0
    visited[start] = (None, 0)  # Mark the start node as visited with no predecessor and cost 0

    while not pq.empty():  # While there are nodes to process
        cost, node = pq.get()  # Dequeue a node with the lowest cost
        if node == end:  # If the end node is reached, break
            break
        for neighbor, connected in enumerate(matrix[node]):  # Iterate over neighbor
            if connected:  # If connected
                new_cost = cost + connected  # Calculate the new cost
                if neighbor not in visited or new_cost < visited[neighbor][1]:  # If not visited or found a cheaper path
                    pq.put((new_cost, neighbor))  # Enqueue the neighbor with the new cost
                    visited[neighbor] = (node, new_cost)  # Mark the neighbor as visited with predecessor and cost

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node][0]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path

def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm 
    heuristic : edge weights
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    openList = PriorityQueue()  # Initialize the priority queue for GBFS
    openList.put((0, start))  # Enqueue the start node with heuristic 0
    visited[start] = None  # Mark the start node as visited with no predecessor
    closedList = []  # Initialize the closed list
    

    while not openList.empty():  # While there are nodes to process
        _, node = openList.get()  # Dequeue a node with the lowest heuristic
        if node == end:  # If the end node is reached, break
            break

        closedList.append(node)  # Add the node to the closed list

        for neighbor, connected in enumerate(matrix[node]):  # Iterate over neighbors
            if connected and neighbor not in closedList:  # If connected and not visited
                heuristic = connected  # Assume that edge weights as heuristic
                openList.put((heuristic, neighbor))  # Enqueue the neighbor with the heuristic
                visited[neighbor] = node  # Mark the neighbor as visited with predecessor

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path

    

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 

    def heuristic(node, end):
        # Calculate the Euclidean distance between the current node and the end node
        return math.sqrt((pos[node][0] - pos[end][0])**2 + (pos[node][1] - pos[end][1])**2)

    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    pq = PriorityQueue()  # Initialize the priority queue for A*
    pq.put((0, start))  # Enqueue the start node with cost 0
    visited[start] = (None, 0)  # Mark the start node as visited with no predecessor and cost 0

    while not pq.empty():  # While there are nodes to process
        _, node = pq.get()  # Dequeue a node with the lowest cost + heuristic
        if node == end:  # If the end node is reached, break
            break
        for neighbor, connected in enumerate(matrix[node]):  # Iterate over neighbors
            if connected:  # If connected
                new_cost = visited[node][1] + connected  # Calculate the new cost
                if neighbor not in visited or new_cost < visited[neighbor][1]:  # If not visited or found a cheaper path
                    total_cost = new_cost + heuristic(neighbor, end)  # Calculate the total cost (cost + heuristic)
                    pq.put((total_cost, neighbor))  # Enqueue the neighbor with the total cost
                    visited[neighbor] = (node, new_cost)  # Mark the neighbor as visited with predecessor and cost

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node][0]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path


def Beam(adj_matrix, start_node, end_node):
    """
    Perform beam search on a graph represented by an adjacency matrix.
    
    Parameters:
    -----------
    adj_matrix : numpy.ndarray
        The adjacency matrix representing the graph. adj_matrix[i][j] is the weight of the edge from node i to node j, or 0 if there is no edge.
    start_node : int
        The index of the starting node.
    end_node : int
        The index of the goal node.
    beam_width : int
        The maximum number of paths to consider at each step.
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    visited = {start_node: None}  # Initialize the visited dictionary
    beam_width = 5 # assume beam width is 2
    pq = PriorityQueue()  # Initialize the priority queue for beam search
    pq.put((0, [start_node]))  # Enqueue the start node with cost 0

    while not pq.empty():  # While there are paths to explore
        current_level = []  # List to store paths at the current level
        for _ in range(min(beam_width, pq.qsize())):  # Expand up to beam_width paths
            cost, path = pq.get()  # Dequeue the path with the lowest cost
            current_node = path[-1]  # Get the current node from the path

            if current_node == end_node:  # If the goal node is reached
                return visited, path  # Return the visited nodes and the path

            for neighbor, weight in enumerate(adj_matrix[current_node]):  # Iterate over neighbors
                if weight > 0 and neighbor not in visited:  # If there is an edge and neighbor is not visited
                    new_cost = cost + weight  # Calculate the new cost
                    new_path = path + [neighbor]  # Create a new path
                    pq.put((new_cost, new_path))  # Enqueue the new path with the new cost
                    visited[neighbor] = current_node  # Mark the neighbor as visited with predecessor

    return visited, []  # Return the visited nodes and an empty path if no path is found