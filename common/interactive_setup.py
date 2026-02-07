import sys
from pathlib import Path
from common.tools.organize_data import organize_classification, organize_paired_data

def setup_dataset_interactive(task_type: str, config_data_dir: Path) -> bool:
    """
    Interactively setup dataset if missing.
    Returns: True if data setup was successful (or user skipped),
             False if setup failed or was aborted.
    """
    print("\n" + "!" * 60)
    print(f"MISSING DATA for {task_type.upper()}")
    print(f"Expected location: {config_data_dir}")
    print("!" * 60)
    
    # Check if TTY or interactive environment
    # Note: sys.stdin.isatty() is often False in Jupyter/Colab, 
    # but input() typically works. We will try input() and catch EOF.
    
    try:
        choice = input(f"\nWould you like to import/organize data now? [y/N]: ").strip().lower()
    except (EOFError, OSError):
        print("[!] Non-interactive session detected. Cannot prompt for data setup.")
        return False
        
    if choice != 'y':
        return False
        
    print("\n--- Interactive Data Setup ---")
    
    # Calculate output root (project/data)
    # config_data_dir e.g. .../data/classification/raw or .../data/segmentation
    # We want .../data
    
    # Find 'data' directory in parents
    data_root = None
    for p in config_data_dir.parents:
        if p.name == 'data':
            data_root = p
            break
            
    if data_root is None:
        # Fallback based on structure heuristic
        if 'classification' in str(config_data_dir):
             # classification/raw -> data
             data_root = config_data_dir.parent.parent
        else:
             # segmentation -> data
             data_root = config_data_dir.parent
    
    # Final Fallback
    if not data_root:
        data_root = config_data_dir.parent
        
    print(f"[*] Target Data Directory: {data_root / task_type}")
    
    try:
        if task_type == 'classification':
            print("Please provide the path to your RAW IMAGES folder.")
            src_str = input("Path: ").strip().strip('"').strip("'")
            if not src_str: return False
            src = Path(src_str)
            if not src.exists():
                print("[!] Path does not exist.")
                return False
                
            print("\nMethod:")
            print(" [1] Filename inference (e.g. cat_01.jpg -> cat)")
            print(" [2] CSV/Excel Metadata")
            
            method = input("Select [1/2]: ").strip()
            
            if method == '1':
                delim = input("Delimiter (default '_'): ").strip() or "_"
                idx = input("Index (default 0): ").strip() or "0"
                organize_classification(
                    src, data_root / task_type,
                    delimiter=delim, index=int(idx)
                )
            elif method == '2':
                csv_str = input("CSV Path: ").strip().strip('"').strip("'")
                if csv_str:
                    organize_classification(
                        src, data_root / task_type,
                        metadata_path=Path(csv_str)
                    )
            else:
                print("[!] Invalid selection.")
                return False
                
        else:
            # Segmentation / Detection
            print("Please provide the path to your IMAGES folder.")
            img_str = input("Images Path: ").strip().strip('"').strip("'")
            print(f"Please provide the path to your {'MASKS' if task_type=='segmentation' else 'LABELS'} folder.")
            lbl_str = input("Labels Path: ").strip().strip('"').strip("'")
            
            if img_str and lbl_str:
                organize_paired_data(
                    Path(img_str), Path(lbl_str),
                    data_root / task_type, task_type
                )
            else:
                return False
                
        print("\n[+] Setup attempted. Verifying...")
        # Check if directory exists and has content
        if config_data_dir.exists() and any(config_data_dir.iterdir()):
            print("✅ Data directory created successfully.")
            return True
        # For classification, check for subdirs
        if task_type == 'classification' and config_data_dir.exists():
             if any(p.is_dir() for p in config_data_dir.iterdir()):
                 return True
                 
        print("⚠️ Data directory still missing/empty.")
        return False
            
    except Exception as e:
        print(f"[!] Setup failed: {e}")
        return False
