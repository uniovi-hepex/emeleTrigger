# OMTFGnnMain.py

import argparse
import os
import yaml
import logging
from OMTFGnnDownloadData import download_data  # Import the download_data function
from OMTFGnnDataProcessing import GraphCreationModel      # Updated import based on the file name
from OMTFGnnOptimize import run_optimization
from OMTFGnnUtils import set_seed, setup_logging                # Ensure this is correctly implemented



def validate_config(config):
    """
    Validates the presence of required sections in the configuration.

    Parameters:
    - config (dict): Loaded YAML configuration.

    Raises:
    - ValueError: If any required section is missing.
    """
    required_sections = ['data_download', 'data_processing', 'training', 'logging', 'resources']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    # Add more detailed checks as needed

def main():
    parser = argparse.ArgumentParser(description="GNN Pipeline with HPO and NAS")
    parser.add_argument("--config", type=str, default="OMTFGnnConfig.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Setup logging
    logs_dir = config['logging'].get('logs_dir', "../logs/")
    setup_logging(logs_dir)
    logging.info("Starting GNN Pipeline...")

    # Validate configuration
    try:
        validate_config(config)
        logging.info("Configuration validation passed.")
    except ValueError as e:
        logging.error(f"Configuration validation failed: {e}")
        exit(1)

    # Set random seed for reproducibility
    set_seed(config['training'].get('seed', 42))
    logging.info(f"Random seed set to {config['training'].get('seed', 42)}")

    # Step 1: Download Data (conditionally)
    if config['data_download'].get('download_data', True):
        logging.info("Starting data download...")
        download_data(config)
        logging.info("Data download completed.")
    else:
        logging.info("Data download step skipped as per configuration.")

    # Step 2: Initialize the GraphCreationModel
    graphs = GraphCreationModel(
        raw_data_dir=config['data_download']['raw_data_dir'],
        graph_save_path=config['data_processing']['graph_save_path'],
        model_connectivity=config['data_processing']['model_connectivity'],
        config=config  # Pass the entire config
    )
    graphs.set_muon_vars(config['data_processing']['muon_vars'])
    graphs.set_stub_vars(config['data_processing']['stub_vars'])
    logging.info("GraphCreationModel initialized.")

    # Step 3: Load and preprocess data
    logging.info("Starting data processing...")
    try:
        graphs.load_data(debug=config['logging'].get('debug', True))
        if hasattr(graphs.dataset, 'empty') and graphs.dataset.empty:
            logging.error("No data loaded. Exiting workflow.")
            exit(1)
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        exit(1)
    logging.info("Data processing completed.")

    # Step 4: Convert to graphs
    logging.info("Converting data to graphs...")
    try:
        graphs.convert_to_graph(
            num_workers=config['resources'].get('num_workers', 4),
            device=config['resources'].get('device', "cuda")
        )
        if not graphs.pyg_graphs:
            logging.error("No graphs were created. Exiting workflow.")
            exit(1)
    except Exception as e:
        logging.error(f"Graph conversion failed: {e}")
        exit(1)
    logging.info("Data conversion to graphs completed.")

    # Step 5: Validate graphs (optional)
    if config['logging'].get('enable_validation', False):
        logging.info("Starting graph validation...")
        graphs.draw_example_graphs(savefig=os.path.join(config['logging']['logs_dir'], "graph_examples.png"))
        graphs.verifyGraphs()
        logging.info("Graph validation completed.")

    # Step 6: Save the dataset
    try:
        graphs.saveTorchDataset()
    except Exception as e:
        logging.error(f"Saving Torch dataset failed: {e}")
        exit(1)
    logging.info("Graph dataset saved.")

    # Step 7: Run Hyperparameter Optimization and NAS
    logging.info("Starting hyperparameter optimization and NAS...")
    try:
        run_optimization(
            graph_creation_model=graphs,
            config=config,
            use_wandb=config['logging'].get('use_wandb', False),
            wandb_project=config['logging'].get('wandb_project', "gnn-optimization-project"),
            wandb_api_key=config['logging'].get('wandb_api_key', None),
            enable_profiling=config['logging'].get('enable_profiling', False)
        )
    except Exception as e:
        logging.error(f"Hyperparameter optimization failed: {e}")
        exit(1)
    logging.info("Hyperparameter optimization and NAS completed.")

    # Step 8: Clean up memory
    graphs.clear_memory()
    logging.info("Memory cleaned up. Workflow completed successfully.")

if __name__ == "__main__":
    main()
