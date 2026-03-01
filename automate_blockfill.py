import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess

DEVICE = "emulator-5554"
ADB_PATH = r"C:\LDPlayer\LDPlayer9\adb.exe"

screen_x1, screen_x2, screen_y1, screen_y2 = 0, 720, 136, 1055
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


def screenshot():
    print("taking screenshot...")
    result = subprocess.run(
        [ADB_PATH, "-s", DEVICE, "exec-out", "screencap", "-p"],
        capture_output=True
    )
    image = np.frombuffer(result.stdout, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def drag(x_start, y_start, x_end, y_end):
    distance = np.hypot(x_end - x_start, y_end - y_start)

    BASE_DURATION = 300
    SPEED_FACTOR  = 1.2
    MAX_DURATION  = 800

    duration_ms = int(BASE_DURATION + distance * SPEED_FACTOR)
    duration_ms = min(duration_ms, MAX_DURATION)

    subprocess.run([
        ADB_PATH, "-s", DEVICE,
        "shell", "input", "swipe",
        str(x_start), str(y_start),
        str(x_end), str(y_end),
        str(duration_ms)
    ])
    time.sleep(duration_ms / 1000 + 0.1)


def tap(next_x, next_y):
    # 42, 42
    # 676, 46

    subprocess.run([
        ADB_PATH, "-s", DEVICE,
        "shell", "input", "tap",
        str(next_x), str(next_y)
    ])


def find_tiles(img):
    print("detecting tiles...")
    # Preprocess image (grayscale, canny, dilate)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grayscale, 50, 150)
    dilated = cv2.dilate(canny, None, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        tiles_data.append([x, y, x+w, y+h, w, h])
    return tiles_data


def find_start_tile(img):
    print("detecting start tile...")
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV Mask
    lower = np.array([0, 50, 100])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        return [x, y, x+w, y+h, w, h]
    else:
        return []


def to_grid(start_tile, tiles_data):
    print("turning tiles into 1&0...")
    tolerate = 5

    # Convert to numpy
    tiles_data = np.array(tiles_data)

    # Mark start tile (vectorized)
    diff = np.abs(tiles_data - start_tile).astype(int)
    mask = np.all(diff <= tolerate, axis=1)
    start_col = mask.astype(int).reshape(-1, 1)
    tiles_data = np.hstack((tiles_data, start_col))

    # Sort top-to-bottom, then left-to-right
    indices = np.lexsort((tiles_data[:, 0], tiles_data[:, 1]))
    tiles_data = tiles_data[indices]

    # Calculate and insert center of tile
    avg_size = np.average(tiles_data[:, 4:6])
    cxy = np.round(tiles_data[:, [0, 1]] + avg_size/2).astype(int)
    center_x = cxy[:, 0]
    center_y = cxy[:, 1] + screen_y1
    tiles_data = np.insert(tiles_data, 2, center_x, axis=1)
    tiles_data = np.insert(tiles_data, 3, center_y, axis=1)

    # Split on index [2, 4, 6, 8] horizontally
    xy1, cxy, xy3, tile_size, start_tile = np.hsplit(tiles_data, [2, 4, 6, 8])

    # Compute grid boundaries (min, max, rows, columns)
    xy_min, xy_max = np.min(xy1, axis=0), np.max(xy3, axis=0)
    x_min, y_min, x_max, y_max = xy_min[0], xy_min[1], xy_max[0], xy_max[1]
    n_row, n_column = int(round((y_max-y_min)/avg_size, 1)), int(round((x_max-x_min)/avg_size, 1))

    # Turn data into 1&0 grid
    grid = np.zeros((n_row, n_column), dtype=int)
    for (x1, y1, cx, cy, x2, y2, w, h, s) in tiles_data:

        col = int(round((x1 - x_min) / avg_size, 1))
        row = int(round((y1 - y_min) / avg_size, 1))

        grid[row, col] = -1 if s else 1

    print("detecting end tile")
    # Find end tile
    neighbors = np.zeros((n_row, n_column), dtype=int)
    for r in range(n_row):
        for c in range(n_column):

            # Skip empty cell & start tile
            if (grid[r][c] == 0) | (grid[r][c] == -1):
                continue

            n_neighbor = 0

            # Check 4 directions
            for move_row, move_col in [(1,0), (-1,0), (0,1), (0,-1)]:
                neighbor_row, neighbor_col = r+move_row, c+move_col

                # Boundary checks
                if 0 <= neighbor_row < n_row and 0 <= neighbor_col < n_column:
                    if grid[neighbor_row][neighbor_col] != 0:
                        n_neighbor += 1

            if n_neighbor == 1:
                grid[r][c] = 2

            neighbors[r][c] = n_neighbor
    return grid, tiles_data


def process_img(img, y1, y2, x1, x2):
    print("processing image...")
    # Format: img[y1:y2, x1:x2]
    cropped = img[y1:y2, x1:x2]
    tiles_data = find_tiles(cropped)
    if tiles_data:
        start_tile = find_start_tile(cropped)
        if start_tile:
            grid, tiles_data = to_grid(start_tile, tiles_data)
            return grid, tiles_data
        else:
            print("Start tile not found.")
    else:
        print("No tiles found.")
        quit()

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
        print("puzzle solved!")
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


def play_puzzle(rows, cols, tiles_data):
    print("playing puzzle...")
    # Split tile data columns
    xy1, cxy, xy2, tile_size, start_tile = np.hsplit(tiles_data, [2, 4, 6, 8])

    # Convert path positions into tile indices
    answer = []

    for r_path, c_path in path:
        n = (r_path + 1) * cols - (cols - (c_path + 1))
        count = n

        for r in range(rows):
            for c in range(cols):
                count -= 1
                if grid[r][c] == 0:
                    n -= 1
                if count <= 0:
                    break
            if count <= 0:
                break
        answer.append(n)
    answer = np.array(answer) - 1

    # Convert tile index -> center coordinates
    correct = cxy[answer]

    # Remove unnecessary straight-line middle points
    tolerate = 7
    result = [correct[0]]
    for i in range(1, len(correct) - 1):
        prev = correct[i - 1]
        curr = correct[i]
        nxt  = correct[i + 1]

        same_x = (
            abs(prev[0] - curr[0]) <= tolerate and
            abs(curr[0] - nxt[0])  <= tolerate
        )

        same_y = (
            abs(prev[1] - curr[1]) <= tolerate and
            abs(curr[1] - nxt[1])  <= tolerate
        )

        if not (same_x or same_y):
            result.append(curr)
    result.append(correct[-1])
    result = np.array(result)

    # Prepare drag start/end coordinates
    offset = int(round(np.average(tile_size)))
    start_points = result.astype(int)
    end_points = np.vstack((
        result[1:],                  # next points
        result[-1] + offset          # last offset move
    )).astype(int)

    # Perform drags
    for (x0, y0), (x1, y1) in zip(start_points, end_points):
        drag(x0, y0, x1, y1)
    print("puzzle completed.")


def reset_state():
    global visited, path, stats

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


def main():
    global grid, rows, cols, ax

    # Turn image into grid
    img = screenshot()
    grid, tiles_data = process_img(img, screen_y1, screen_y2, screen_x1, screen_x2)
    
    rows, cols = grid.shape
    start_tile = tuple(np.argwhere(grid == -1)[0])

    plt.ion()
    fig, ax = plt.subplots()
    
    # Solve puzzle (depth first search)
    print("solving puzzle...")
    start_time = time.time()
    dfs(start_tile[0], start_tile[1], 1)
    end_time = time.time()
    stats["time"] = end_time - start_time

    plt.close("all")

    # Report
    print("\n===== REPORT =====")
    print(f"Tiles in solution     : {len(path)}")
    print(f"Nodes explored        : {stats['nodes']}")
    print(f"Branches attempted    : {stats['branches']}")
    print(f"Backtracks            : {stats['backtracks']}")
    print(f"Branches pruned       : {stats['pruned']}")
    print(f"Max depth reached     : {stats['max_depth']}")
    print(f"Time taken            : {format_time(stats['time'])}\n")

    play_puzzle(rows, cols, tiles_data)
    print("\n\n\n")


if __name__ == "__main__":
    to_solve = 3
    for lvl in range(to_solve):
        print("next level.")
        reset_state()
        main()
        time.sleep(7)