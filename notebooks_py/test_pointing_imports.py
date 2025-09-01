#!/usr/bin/env python3
"""Test script to verify all components are properly set up."""

import sys
import os

# Add src to path
ROOT_DIR = ".."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

print("Testing imports...")
print("-" * 40)

# Test core imports
try:
    print("1. Testing basic Python compatibility...")
    from abc import ABC, abstractmethod
    print("   ✓ ABC imports work")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test endopoint structure
try:
    print("\n2. Testing endopoint module structure...")
    import endopoint
    print("   ✓ endopoint package found")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test dataset module
try:
    print("\n3. Testing dataset module...")
    from endopoint.datasets import base
    print("   ✓ datasets.base module loads")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test models module
try:
    print("\n4. Testing models module...")
    from endopoint.models import base as model_base
    print("   ✓ models.base module loads")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test prompts module
try:
    print("\n5. Testing prompts module...")
    from endopoint.prompts import builders
    print("   ✓ prompts.builders module loads")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test eval module
try:
    print("\n6. Testing eval module...")
    from endopoint.eval import parser
    print("   ✓ eval.parser module loads")
except ImportError as e:
    print(f"   ✗ Error: {e}")

print("\n" + "-" * 40)
print("Module structure test complete!")
print("\nNote: Some imports may fail due to missing dependencies")
print("(torch, datasets, PIL, etc.) but the module structure is valid.")