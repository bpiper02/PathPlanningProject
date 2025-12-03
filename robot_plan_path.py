import argparse
from mbot_bridge.api import MBot

from src.graph import GridGraph, Cell
from src.graph_search import a_star_search
from src.utils import generate_plan_file


def convert_cell_path_to_poses(cell_path, graph):
    """Converts a list of Cell objects to robot pose coordinates.
    
    Args:
        cell_path: List of Cell objects representing the path
        graph: GridGraph instance for coordinate conversion
        
    Returns:
        List of [x, y, theta] pose tuples for robot navigation
    """
    return [[*graph.cell_to_pos(cell.i, cell.j), 0] for cell in cell_path]


def parse_command_line_args():
    """Parses command line arguments for robot path planning.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Path planning for MBot robot using A* search algorithm."
    )
    parser.add_argument(
        "-m", "--map", 
        type=str, 
        default="/home/mbot/current.map",
        help="Path to the occupancy grid map file"
    )
    parser.add_argument(
        "--goal", 
        type=float, 
        nargs=2, 
        default=[0, 0],
        help="Goal position in meters (x, y)"
    )
    parser.add_argument(
        "-r", "--collision-radius", 
        type=float, 
        default=0.15,
        help="Robot collision radius in meters"
    )

    return parser.parse_args()


def main():
    """Main execution function for robot path planning."""
    args = parse_command_line_args()

    # Initialize occupancy grid from map file
    occupancy_grid = GridGraph(
        args.map, 
        collision_radius=args.collision_radius
    )
    
    # Convert goal position to grid cell coordinates
    goal_cell = occupancy_grid.pos_to_cell(*args.goal)

    # Initialize robot interface and get current pose
    robot_interface = MBot()
    current_robot_pose = robot_interface.read_slam_pose()
    start_cell = occupancy_grid.pos_to_cell(*current_robot_pose[:2])

    # Compute optimal path using A* search
    planned_path = a_star_search(occupancy_grid, start_cell, goal_cell)

    # Execute path on robot
    if planned_path:
        print(f"Found path with {len(planned_path)} waypoints. Executing navigation...")
        robot_poses = convert_cell_path_to_poses(planned_path, occupancy_grid)
        robot_interface.drive_path(robot_poses)
        
        # Generate visualization file
        generate_plan_file(occupancy_grid, start_cell, goal_cell, planned_path, algo="astar")
    else:
        print("No valid path found to goal!")


if __name__ == "__main__":
    main()

