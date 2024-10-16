import numpy as np
from queue import Queue, PriorityQueue
import math

def BFS(matrix, start, end):
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
    path = []  # Initialize the path list
    visited = {}  # Initialize the visited dictionary
    pq = PriorityQueue()  # Initialize the priority queue for GBFS
    pq.put((0, start))  # Enqueue the start node with heuristic 0
    visited[start] = None  # Mark the start node as visited with no predecessor

    while not pq.empty():  # While there are nodes to process
        _, node = pq.get()  # Dequeue a node with the lowest heuristic
        if node == end:  # If the end node is reached, break
            break
        for neighbor, connected in enumerate(matrix[node]):  # Iterate over neighbors
            if connected and neighbor not in visited:  # If connected and not visited
                heuristic = connected  # Assume that edge weights as heuristic
                pq.put((heuristic, neighbor))  # Enqueue the neighbor with the heuristic
                visited[neighbor] = node  # Mark the neighbor as visited with predecessor

    if end in visited:  # If the end node was reached
        node = end
        while node is not None:  # Trace back the path from end to start
            path.insert(0, node)  # Insert the node at the beginning of the path
            node = visited[node]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the path

def Astar(matrix, start, end, pos):
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