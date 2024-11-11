# OMTFGnnDownloadData.py

import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import logging

def download_file(url, dest_path):
    """
    Downloads a file from a URL to a destination path with a progress bar.

    Parameters:
    - url (str): URL of the file to download.
    - dest_path (str): Path to save the downloaded file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        logging.info(f"Successfully downloaded {os.path.basename(dest_path)}")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while downloading {url}: {http_err}")
    except Exception as err:
        logging.error(f"Error occurred while downloading {url}: {err}")

def get_file_links(base_url):
    """
    Scrapes the CERNBox page to get downloadable file links.

    Parameters:
    - base_url (str): Base URL of the CERNBox directory.

    Returns:
    - list of str: List of downloadable file URLs.
    """
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        file_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and 'download' in href:
                # Construct the full download URL
                file_links.append(href)
        logging.info(f"Found {len(file_links)} files to download.")
        return file_links
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while accessing {base_url}: {http_err}")
        return []
    except Exception as err:
        logging.error(f"Error occurred while accessing {base_url}: {err}")
        return []

def download_data(config):
    """
    Downloads ROOT files based on the configuration.

    Parameters:
    - config (dict): Loaded YAML configuration.
    """
    data_base_url = config['data_download']['data_base_url']
    file_prefix = config['data_download']['file_prefix']
    file_suffix = config['data_download']['file_suffix']
    start_index = config['data_download']['start_index']
    end_index = config['data_download']['end_index']
    raw_data_dir = config['data_download']['raw_data_dir']
    os.makedirs(raw_data_dir, exist_ok=True)

    # Generate file names based on the prefix, suffix, and index range
    file_names = [f"{file_prefix}{i}{file_suffix}" for i in range(start_index, end_index + 1)]

    # Download each file
    for file_name in file_names:
        url = data_base_url + file_name
        dest_path = os.path.join(raw_data_dir, file_name)

        if not os.path.exists(dest_path):
            logging.info(f"Starting download of {file_name}")
            download_file(url, dest_path)
        else:
            logging.info(f"{file_name} already exists. Skipping download.")
