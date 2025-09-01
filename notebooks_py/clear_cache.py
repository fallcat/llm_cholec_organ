#!/usr/bin/env python
"""Clear the model cache for endopoint."""

import shutil
from pathlib import Path

def clear_cache():
    """Clear the endopoint model cache."""
    cache_dir = Path.home() / ".cache" / "endopoint" / "models"
    
    if cache_dir.exists():
        print(f"🗑️  Clearing cache at: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleared successfully!")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print(f"ℹ️  Cache directory doesn't exist: {cache_dir}")
    
    print("\nYou can now run the evaluation with fresh API calls.")
    print("Note: This will use API credits as responses won't be cached.")

if __name__ == "__main__":
    clear_cache()