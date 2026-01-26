from cognee.shared.utils import start_visualization_server


def visualization_server(port, host=None):
    """
    Start a visualization server on the specified port.

    Args:
        port (int): The port number to run the server on
        host (str, optional): Host/IP to bind. Defaults to utils default.

    Returns:
        callable: A shutdown function that can be called to stop the server

    Raises:
        ValueError: If port is not a valid port number
    """
    kwargs = {"port": port}
    if host:
        kwargs["host"] = host
    return start_visualization_server(**kwargs)
