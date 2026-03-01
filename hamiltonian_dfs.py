import numpy as np
import time
import matplotlib.pyplot as plt


visited = set()
path = []
stats = {
    "nodes": 0,
    "branches": 0,
    "backtracks": 0,
    "pruned": 0,
    "max_depth": 0,
    "time": 0
}


def draw():
    rows, cols = grid.shape
    ax.clear()
    display = np.zeros((rows, cols, 3))

    # Build quick lookup for path order (O(1))
    path_index = {pos: i for i, pos in enumerate(path)}

    for r in range(rows):
        for c in range(cols):

            # Base colors
            if grid[r][c] == 0:
                display[r, c] = [0.1, 0.1, 0.1]      # dark gray
            elif grid[r][c] == 1:
                display[r, c] = [0.9, 0.9, 0.9]      # light gray
            elif grid[r][c] == -1:
                display[r, c] = [1, 0.3, 0.3]        # bright red start
            elif grid[r][c] == 2:
                display[r, c] = [0.2, 0.9, 0.2]      # bright green end

            # If part of path
            if (r, c) in path_index:
                idx = path_index[(r, c)]
                t = idx / max(len(path), 1)

                # Blue gradient
                display[r, c] = [0.1, 0.4 + 0.6*t, 1]
    ax.imshow(display)

    # Grid lines
    ax.set_xticks(np.arange(-.5, cols, 1))
    ax.set_yticks(np.arange(-.5, rows, 1))
    ax.grid(color='white', linestyle='-', linewidth=1.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add numbers on tiles (visit order)
    for (r, c), idx in path_index.items():
        ax.text(
            c, r,
            str(idx+1),
            ha='center',
            va='center',
            fontsize=8,
            color='black',
            weight='bold'
        )

    # Overlay live stats
    ax.set_title(
        f"Nodes: {stats['nodes']} | "
        f"Branches: {stats['branches']} | "
        f"Pruned: {stats['pruned']} | "
        f"Depth: {stats['max_depth']}",
        fontsize=10
    )
    plt.pause(0.01)


def degree(r, c):
    count = 0
    for move_row, move_column in [(1,0), (-1,0), (0,1), (0,-1)]:
        neighbor_r, neighbor_c = r+move_row, c+move_column
        if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols:
            if grid[neighbor_r][neighbor_c] != 0 and (neighbor_r,neighbor_c) not in visited:
                count += 1
    return count


def remaining_connected():
    remaining = [(r, c) for r in range(rows)
                           for c in range(cols)
                           if grid[r][c] != 0 and (r,c) not in visited]

    if not remaining:
        return True

    # BFS from first remaining tile
    stack = [remaining[0]]
    seen = set([remaining[0]])

    while stack:
        r, c = stack.pop()
        for move_row, move_column in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor_r, neighbor_c = r+move_row, c+move_column
            if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols:
                if grid[neighbor_r][neighbor_c] != 0 and (neighbor_r,neighbor_c) not in visited and (neighbor_r,neighbor_c) not in seen:
                    seen.add((neighbor_r,neighbor_c))
                    stack.append((neighbor_r,neighbor_c))

    return len(seen) == len(remaining)


def forced_move_violation():
    forced_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r,c) not in visited and grid[r][c] != 2:
                count = 0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc] != 0 and (nr,nc) not in visited:
                            count += 1
                if count == 1:
                    forced_count += 1
                if count == 0:
                    return True  # immediate failure

    return forced_count > 2


def dfs(r, c, count):
    global visited, path
    total_tiles = np.sum(grid != 0)

    # Boundary check
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False

    # Blocked / Visited
    if grid[r][c] == 0 or (r,c) in visited:
        return False

    # Reached end too early
    if grid[r][c] == 2 and count != total_tiles:
        return False

    # Update path
    visited.add((r,c))
    path.append((r,c))
    stats["nodes"] += 1
    stats["max_depth"] = max(stats["max_depth"], count)
    draw()

    # Check win condition
    if count == total_tiles and grid[r][c] == 2:
        return True

    # Dead-end pruning
    if not remaining_connected() or forced_move_violation():
        visited.remove((r,c))
        path.pop()
        stats["pruned"] += 1
        draw()
        return False

    # Generate next move
    neighbors = []
    for move_row, move_column in [(1,0), (-1,0), (0,1), (0,-1)]:
        neighbor_r, neighbor_c = r+move_row, c+move_column
        if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols:
            if grid[neighbor_r][neighbor_c] != 0 and (neighbor_r, neighbor_c) not in visited:
                neighbors.append((neighbor_r, neighbor_c))

    # Sort by least available moves first
    neighbors.sort(key=lambda pos: degree(pos[0], pos[1]))

    # Explore each branch (recursive step)
    for neighbor_r, neighbor_c in neighbors:
        stats["branches"] += 1
        if dfs(neighbor_r, neighbor_c, count+1):
            return True

    # Backtracking
    visited.remove((r,c))
    path.pop()
    stats["backtracks"] += 1
    draw()
    return False
    

def format_time(sec):
    # Format remaining time
    hours = round(sec) // 3600
    minutes = (round(sec) % 3600) // 60
    seconds = round(sec) % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours:01}h")
    if minutes > 0:
        parts.append(f"{minutes:01}m")
    if seconds > 0:
        parts.append(f"{seconds:01}s")

    time_str = " ".join(parts)
    return time_str


def main():
    global grid, rows, cols, ax
    
    # start: -1, end: 2, wall: 0, tile: 1
    grid = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  2,  1,  1,  0,  1,  1,  0,  0],
            [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
            [ 0,  0,  1,  1, -1,  1,  1,  1,  0,  0],
            [ 0,  0,  0,  1,  1,  1,  1,  1,  0,  0],
            [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
            [ 0,  0,  1,  1,  1,  1,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
    
    grid = np.array(grid)
    rows, cols = grid.shape
    start_tile = tuple(np.argwhere(grid == -1)[0])

    plt.ion()
    fig, ax = plt.subplots()
    
    # Solve puzzle (depth first search)
    start_time = time.time()
    dfs(start_tile[0], start_tile[1], 1)
    end_time = time.time()
    stats["time"] = end_time - start_time

    if len(path) == 0:
        print("No solution found.")
        plt.close("all")

    # Report
    print("\n===== REPORT =====")
    print(f"Tiles in solution     : {len(path)}")
    print(f"Nodes explored        : {stats['nodes']}")
    print(f"Branches attempted    : {stats['branches']}")
    print(f"Backtracks            : {stats['backtracks']}")
    print(f"Branches pruned       : {stats['pruned']}")
    print(f"Max depth reached     : {stats['max_depth']}")
    print(f"Time taken            : {format_time(stats["time"])}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()