#!/usr/bin/env python3
"""
Script to extract the last configuration from ASE .merged.traj files
and save them in LAMMPS data format.

This script:
1. Finds all p* folders in the current directory
2. Looks for NPT_ASE/*/analysis/*.merged.traj files
3. Reads the last configuration (index=-1) from each trajectory
4. Saves it as data_liquid.txt in the same analysis folder
"""

import os
import glob
from pathlib import Path
from ase.io import read, write


def find_p_folders(base_dir):
    """Find all folders matching p* pattern."""
    p_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('p') and item[1:]:
            try:
                float(item[1:])
                p_folders.append(item_path)
            except ValueError:
                continue
            #p_folders.append(item_path)
    return sorted(p_folders)


def find_merged_traj_files(p_folder):
    """Find all .merged.traj files in NPT_ASE/*/analysis/ directories."""
    pattern = os.path.join(p_folder, 'liquid', 'NPT_ASE', '*', 'analysis', '*.merged.traj')
    traj_files = glob.glob(pattern)
    return traj_files


def extract_and_save_last_config(traj_file):
    """Extract last configuration from trajectory and save as LAMMPS data file."""
    try:
        # Read the last configuration (index=-1)
        atoms = read(traj_file, index='-1')
        
        # Determine output file path (same directory as trajectory)
        analysis_dir = os.path.dirname(traj_file)
        output_file = os.path.join(analysis_dir, 'last_config_liquid.txt')
        
        # Write in LAMMPS data format
        # Use format='lammps-data' which writes in the standard LAMMPS data format
        write(output_file, atoms, format='lammps-data')
        
        print(f"✓ Extracted last config from {traj_file}")
        print(f"  Saved to: {output_file}")
        print(f"  Number of atoms: {len(atoms)}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {traj_file}: {str(e)}")
        return False


def main():
    # Get the base directory (parent of script location or current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    
    print(f"Searching for p* folders in: {base_dir}")
    print("-" * 80)
    
    # Find all p* folders
    p_folders = find_p_folders(base_dir)
    
    if not p_folders:
        print(f"No p* folders found in {base_dir}")
        return
    
    print(f"Found {len(p_folders)} p* folder(s)")
    print("-" * 80)
    
    total_traj_files = 0
    total_success = 0
    total_failed = 0
    
    # Process each p* folder
    for p_folder in p_folders:
        print(f"\nProcessing: {os.path.basename(p_folder)}")
        
        # Find all .merged.traj files in this p* folder
        traj_files = find_merged_traj_files(p_folder)
        
        if not traj_files:
            print(f"  No .merged.traj files found in {p_folder}/liquid/NPT_ASE/*/analysis/")
            continue
        
        print(f"  Found {len(traj_files)} .merged.traj file(s)")
        
        # Process each trajectory file
        for traj_file in traj_files:
            total_traj_files += 1
            if extract_and_save_last_config(traj_file):
                total_success += 1
            else:
                total_failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trajectory files found: {total_traj_files}")
    print(f"Successfully processed: {total_success}")
    print(f"Failed: {total_failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()

