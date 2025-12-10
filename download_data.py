#!/usr/bin/env python3
"""
Download the Tiny Shakespeare dataset.

This dataset contains ~1MB of Shakespeare's works concatenated together.
It's small enough to train on a laptop but complex enough to learn patterns.

Source: Karpathy's char-rnn repository
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "input.txt")


def download_shakespeare():
    """Download Tiny Shakespeare dataset if not present."""
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(DATA_FILE):
        print(f"Dataset already exists at {DATA_FILE}")
        with open(DATA_FILE, 'r') as f:
            text = f.read()
        print(f"  Size: {len(text):,} characters")
        print(f"  Unique characters: {len(set(text))}")
        return
    
    print(f"Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    
    # Verify download
    with open(DATA_FILE, 'r') as f:
        text = f.read()
    
    print(f"Downloaded to {DATA_FILE}")
    print(f"  Size: {len(text):,} characters")
    print(f"  Unique characters: {len(set(text))}")
    print(f"\nSample:")
    print("-" * 50)
    print(text[:500])
    print("-" * 50)


if __name__ == '__main__':
    download_shakespeare()
