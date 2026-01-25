import glob
import os
import re
import sys
from collections import defaultdict

def get_latest_log_file(log_dir="logs"):
    """Finds the most recently modified .log file in the log_dir."""
    if not os.path.exists(log_dir):
        return None
    list_of_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def parse_log_line(line):
    """Parses a structlog line into a dictionary."""
    data = {}
    
    # Extract timestamp (ISO 8601ish)
    timestamp_match = re.search(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', line)
    if timestamp_match:
        data['timestamp'] = timestamp_match.group(1)

    # Identify event type
    if "span_start" in line:
        data['event'] = 'start'
    elif "span_end" in line:
        data['event'] = 'end'
    else:
        return None

    # Extract key-value pairs (simple regex, robust enough for structlog default format)
    # Pattern: key=value (no quotes usually in this setup, or simplistic)
    pairs = re.findall(r'(\w+)=([^\s]+)', line)
    for key, value in pairs:
        data[key] = value
        
    return data

def build_trace_tree(events):
    """Groups events by correlation_id."""
    traces = defaultdict(list)
    for e in events:
        if 'correlation_id' in e:
            traces[e['correlation_id']].append(e)
    return traces

def print_trace(correlation_id, events):
    """Reconstructs and prints the trace tree from a flat list of events."""
    print(f"\n{'='*100}")
    print(f"TRACE ID: {correlation_id}")
    print(f"{'='*100}")
    print(f"{'Total (ms)':<12} {'Self (ms)':<12} {'Operation'}")
    print(f"{'-'*12} {'-'*12} {'-'*60}")

    # Sort events by time
    events.sort(key=lambda x: x.get('timestamp', ''))

    # Helper class for Tree Nodes
    class Node:
        def __init__(self, name, start_ts, depth):
            self.name = name
            self.start_ts = start_ts
            self.depth = depth
            self.duration = 0.0
            self.children = []
            self.parent = None

    # Root nodes (depth wise, top level)
    # Since we can have partial traces or multiple roots if context reset, 
    # we just track active usage.
    
    # A simplified stack approach:
    # When 'start', push to stack.
    # When 'end', pop from stack.
    # The logs have 'depth' info, which helps verify.
    
    stack = []
    completed_nodes = []
    
    for e in events:
        if e['event'] == 'start':
            node = Node(e.get('span_name'), e.get('timestamp'), int(e.get('depth', 0)))
            if stack:
                parent = stack[-1]
                # Sanity check depth
                if node.depth > parent.depth:
                    parent.children.append(node)
                    node.parent = parent
                else:
                    # Async or weird nesting? Treat as root or sibling if stack logic fails
                    # But TraceSpan ensures strictly nested ContextVars.
                    pass
            stack.append(node)
            
        elif e['event'] == 'end':
            # find matching node in stack (reverse search)
            name = e.get('span_name')
            depth = int(e.get('depth', 0))
            duration = float(e.get('duration_ms', 0))
            
            # Pop until we find the match
            # In a perfectly nested sync world, it's stack[-1].
            # In async, we might see interleaved starts, but "end" should usually close the most recent open span of that depth?
            # Actually with contextvars, the logging happens in context.
            
            # Simple matcher: look for last node in stack with same name and depth
            matched_node = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i].name == name and stack[i].depth == depth:
                    matched_node = stack.pop(i)
                    break
            
            if matched_node:
                matched_node.duration = duration
                if matched_node.parent is None:
                    completed_nodes.append(matched_node)

    # Recursive print
    def print_node(node, indent_str=""):
        # Calculate self time = duration - sum(children durations)
        children_duration_sum = sum(c.duration for c in node.children)
        self_time = node.duration - children_duration_sum
        if self_time < 0: self_time = 0 # Clock skew or rounding quirks

        # Tree connectors
        connector = "├─ " if node.children else "└─ " # Simplified visualization
        
        print(f"{node.duration:10.2f}   {self_time:10.2f}   {indent_str}{node.name}")
        
        for i, child in enumerate(node.children):
            print_node(child, indent_str + "  ")

    for node in completed_nodes:
        print_node(node)

def analyze_log_file(filepath):
    print(f"Reading log file: {filepath}")
    
    events = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed:
                    events.append(parsed)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    traces = build_trace_tree(events)
    
    if not traces:
        print("No traces found.")
        return

    # Sort traces by time (using start time of first event)
    sorted_traces = sorted(traces.items(), key=lambda x: x[1][0].get('timestamp', '') if x[1] else '')

    # Print the last few traces
    print(f"Found {len(sorted_traces)} traces. Showing the last 2:")
    for cid, evts in sorted_traces[-2:]:
        print_trace(cid, evts)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = get_latest_log_file()
    
    if target_file:
        analyze_log_file(target_file)
    else:
        print("Could not find any log files in logs/")
