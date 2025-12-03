import numpy as np
import heapq
from .graph import Cell
from .utils import trace_path

"""
General graph search instructions:

First, define the correct data type to keep track of your visited cells
and add the start cell to it. If you need to initialize any properties
of the start cell, do that too.

Next, implement the graph search function. When you find a path, use the
trace_path() function to return a path given the goal cell and the graph. You
must have kept track of the parent of each node correctly and have implemented
the graph.get_parent() function for this to work. If you do not find a path,
return an empty list.

To visualize which cells are visited in the navigation webapp, save each
visited cell in the list in the graph class as follows:
     graph.visited_cells.append(Cell(cell_i, cell_j))
where cell_i and cell_j are the cell indices of the visited cell you want to
visualize.
"""


def a_star_search(graph, start, goal):
    """A* Search algorithm.
    
    Uses a priority queue with f(n) = g(n) + h(n) where:
    - g(n): actual cost from start to node n
    - h(n): heuristic estimate from node n to goal (Euclidean distance)
    
    Guarantees optimal path when heuristic is admissible.
    """
    graph.init_graph()
    
    # Heuristic function: Euclidean distance to goal
    def heuristic_estimate(cell):
        dx = cell.i - goal.i
        dy = cell.j - goal.j
        return np.sqrt(dx * dx + dy * dy)
    
    # Priority queue: (f_score, g_score, tiebreaker, cell)
    # Using tiebreaker counter to ensure stable ordering
    tiebreaker_counter = 0
    priority_queue = [(heuristic_estimate(start), 0, tiebreaker_counter, start)]
    
    # Track closed/visited nodes
    closed_set = set()
    
    # Initialize start node
    start_key = (start.i, start.j)
    graph.cost_map[start_key] = 0
    graph.parent_map[start_key] = None
    
    # Main search loop
    while priority_queue:
        f_score, g_score, _, current_node = heapq.heappop(priority_queue)
        node_key = (current_node.i, current_node.j)
        
        # Skip if already processed
        if node_key in closed_set:
            continue
        
        # Mark as closed
        closed_set.add(node_key)
        graph.visited_cells.append(Cell(current_node.i, current_node.j))
        
        # Check if we reached the goal
        if current_node.i == goal.i and current_node.j == goal.j:
            return trace_path(goal, graph)
        
        # Explore neighbors
        adjacent_cells = graph.find_neighbors(current_node.i, current_node.j)
        for next_cell in adjacent_cells:
            neighbor_key = (next_cell.i, next_cell.j)
            
            # Skip if already processed
            if neighbor_key in closed_set:
                continue
            
            # Calculate new cost from start to neighbor
            tentative_g_score = g_score + 1
            
            # Update if we found a better path to this neighbor
            if neighbor_key not in graph.cost_map or tentative_g_score < graph.cost_map[neighbor_key]:
                graph.cost_map[neighbor_key] = tentative_g_score
                graph.parent_map[neighbor_key] = Cell(current_node.i, current_node.j)
                
                # Calculate f_score and add to priority queue
                h_score = heuristic_estimate(next_cell)
                f_score = tentative_g_score + h_score
                tiebreaker_counter += 1
                heapq.heappush(priority_queue, (f_score, tentative_g_score, tiebreaker_counter, next_cell))
    
    # No path found
    return []
