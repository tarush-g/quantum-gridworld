def read_layout(path):
    VALID_CHARS = {"#", "A", "G", "."}

    obstacles = set()
    start = None
    goal = None
    width = None

    try:
        with open(path, "r") as f:
            rows = [line.strip() for line in f]
    except FileNotFoundError:
        raise FileNotFoundError()
    except OSError as e:
        raise OSError()

    # Ignore fully blank lines
    rows = [r for r in rows if r != ""]

    if not rows:
        raise ValueError(f"File is empty: {path}")

    for row, line in enumerate(rows):
        if width is None:
            width = len(line)
        elif len(line) != width:
            raise ValueError(f"Row {row} has {len(line)} characters, expected {width} ")

        for col, cell in enumerate(line):
            if cell not in VALID_CHARS:
                raise ValueError(f"Invalid character {cell!r} at row {row}, col {col}")
            if cell == "#":
                obstacles.add((row, col))
            elif cell == "A":
                if start is not None:
                    raise ValueError(f"Multiple start positions found (row {row}, col {col})")
                start = (row, col)
            elif cell == "G":
                if goal is not None:
                    raise ValueError(f"Multiple goal positions found (row {row}, col {col})")
                goal = (row, col)

    if start is None:
        raise ValueError("No start position 'A' found")
    if goal is None:
        raise ValueError("No goal position 'G' found")

    return start, goal, obstacles