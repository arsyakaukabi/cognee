import argparse
import os
from pathlib import Path

from cognee.api.v1.visualize.start_visualization_server import visualization_server


def main():
    default_port = int(os.getenv("VISUALIZATION_PORT", "9000"))

    parser = argparse.ArgumentParser(
        description="Serve the generated graph visualization HTML on a chosen port."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to serve on (default: env VISUALIZATION_PORT or {default_port})",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "cognee"
            / ".data_storage"
            / "visualizations"
            / "graph_visualization.html"
        ),
        help="Path to the graph_visualization.html file.",
    )
    args = parser.parse_args()

    html_path = Path(args.path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"Visualization file not found: {html_path}")

    # Serve the directory containing the HTML file
    os.chdir(html_path.parent)
    shutdown = visualization_server(port=args.port)
    print(
        f"Serving {html_path.name} at http://localhost:{args.port}/{html_path.name}\n"
        "Press Ctrl+C to stop."
    )
    try:
        # Keep the process alive until interrupted
        import signal

        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
