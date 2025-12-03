import argparse
from src.graph import GridGraph, Cell
from src.graph_search import a_star_search
from src.utils import generate_plan_file


def parse_command_line_args():
    """Parses command line arguments for path planning CLI.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Path planning visualization tool for grid-based maps."
    )
    parser.add_argument(
        "-m", "--map", 
        type=str, 
        required=True,
        help="Path to the occupancy grid map file"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        nargs=2, 
        required=True,
        help="Start cell coordinates (i, j)"
    )
    parser.add_argument(
        "--goal", 
        type=int, 
        nargs=2, 
        required=True,
        help="Goal cell coordinates (i, j)"
    )

    return parser.parse_args()


def run_path_planning(map_file, start_coords, goal_coords):
    """Executes A* path planning and generates visualization file.
    
    Args:
        map_file: Path to the map file
        start_coords: Tuple of (i, j) start cell coordinates
        goal_coords: Tuple of (i, j) goal cell coordinates
        
    Returns:
        List of Cell objects representing the path, or empty list if no path found
    """
    # Initialize occupancy grid
    occupancy_grid = GridGraph(map_file)
    
    # Create start and goal cell objects
    start_cell = Cell(*start_coords)
    goal_cell = Cell(*goal_coords)
    
    # Execute A* path planning
    planned_path = a_star_search(occupancy_grid, start_cell, goal_cell)
    
    # Generate visualization file
    generate_plan_file(occupancy_grid, start_cell, goal_cell, planned_path, algo="astar")
    
    return planned_path


def main():
    """Main execution function for path planning CLI."""
    args = parse_command_line_args()
    
    planned_path = run_path_planning(
        args.map,
        tuple(args.start),
        tuple(args.goal)
    )
    
    if planned_path:
        print(f"Path found with {len(planned_path)} waypoints using A* search.")
    else:
        print("No path found between start and goal.")


if __name__ == "__main__":
    main()
