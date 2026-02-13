def render_progress(done: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {done}/{total} ({done/total:.0%})"

