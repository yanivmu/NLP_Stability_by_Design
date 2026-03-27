#!/bin/bash

# Configuration: Where to look for slurm files
SEARCH_DIR="./scripts/slurm/phase_3"

# Check if the directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR not found."
    exit 1
fi

echo "Searching for .slurm files in $SEARCH_DIR..."

# Find all .slurm files recursively and loop through them
# Using -type f to only find files
find "$SEARCH_DIR" -type f -name "*.slurm" | while read -r slurm_file; do
    echo "Submitting: $slurm_file"
    sbatch "$slurm_file"
    sleep 0.1  # Add a tiny sleep to avoid overwhelming the scheduler 
    
done

echo "Done."
echo "Using 'squeue --me' to check your job status."
squeue --me