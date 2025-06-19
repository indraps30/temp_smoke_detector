# Import required libraries.
import yaml
import joblib
import logging

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s | %(asctime)s | %(message)s")

CONFIG_DIR = "config/config.yaml"

def load_config():
    """Load the configuration file."""
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise Exception(f"Configuration file is not found in path\nError: {error}")
    
    return config

def pickle_load(file_path):
    """Load and serialized file."""
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """Dump data into pickle file."""
    joblib.dump(data, file_path)
    
    
params = load_config()
PRINT_DEBUG = params['print_debug']

def print_debug(message):
    """Check whether user wants to use debug or not."""
    if PRINT_DEBUG:
        logging.info(message)