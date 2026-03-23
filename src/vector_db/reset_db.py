import os
import shutil
import time
import gc


def reset_db(chroma_path: str, logger=None):
    if os.path.exists(chroma_path):
        gc.collect()  # Force garbage collection to release file handles
        time.sleep(1)  # Ensure all file handles are released before deletion
        shutil.rmtree(chroma_path)
        
        
        logger.info(f"Database at '{chroma_path}' has been reset.")
        print(f"Database at '{chroma_path}' has been reset.")
    else:
        logger.info(f"No database found at '{chroma_path}' to reset.")
        print(f"No database found at '{chroma_path}' to reset.")
