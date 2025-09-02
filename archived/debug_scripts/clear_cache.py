#!/usr/bin/env python
"""Clear the model cache for endopoint and old llms.py cache."""

import shutil
from pathlib import Path

def clear_cache():
    """Clear ALL model caches."""
    
    caches_cleared = 0
    
    # 1. New endopoint cache
    cache_dir = Path.home() / ".cache" / "endopoint" / "models"
    if cache_dir.exists():
        print(f"🗑️  Clearing NEW cache at: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ New cache cleared successfully!")
            caches_cleared += 1
        except Exception as e:
            print(f"❌ Error clearing new cache: {e}")
    else:
        print(f"ℹ️  New cache directory doesn't exist: {cache_dir}")
    
    # 2. Old llms.py cache
    old_cache = Path("..") / "src" / ".llms.py.cache"
    if old_cache.exists():
        print(f"🗑️  Clearing OLD cache at: {old_cache}")
        try:
            shutil.rmtree(old_cache)
            print("✅ Old cache cleared successfully!")
            caches_cleared += 1
        except Exception as e:
            print(f"❌ Error clearing old cache: {e}")
    else:
        print(f"ℹ️  Old cache directory doesn't exist: {old_cache}")
    
    print("\n" + "="*50)
    if caches_cleared > 0:
        print(f"✅ Cleared {caches_cleared} cache(s) successfully!")
        print("\nYou can now run the evaluation with fresh API calls.")
        print("Note: This will use API credits as responses won't be cached.")
    else:
        print("ℹ️  No caches found to clear - already clean!")
    
    print("\n🔍 To verify few-shot is working differently from zero-shot:")
    print("   EVAL_USE_CACHE=false EVAL_QUICK_TEST=true python3 eval_pointing.py")
    print("\nIf results are STILL the same, the issue is NOT cache-related.")

if __name__ == "__main__":
    clear_cache()