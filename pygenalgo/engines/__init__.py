import logging

# Create a logger object for the engine module.
logger = logging.getLogger("engines_logger")

# Setup a basic configuration.
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

# Public interface.
__all__ = ["logger"]

