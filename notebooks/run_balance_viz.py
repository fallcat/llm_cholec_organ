#!/usr/bin/env python3
"""Run the balance visualization in a notebook-friendly way"""

import sys
sys.path.append('/shared_data0/weiqiuy/llm_cholec_organ/notebooks')

from visualize_dataset_balancing import create_balance_visualizations, create_coverage_matrix_visualization

# Create comprehensive balance visualizations
print("📊 Creating balance visualizations for CholecSeg8k...")
fig1 = create_balance_visualizations("cholecseg8k_local")

print("\n📊 Creating coverage matrix visualization...")
fig2 = create_coverage_matrix_visualization("cholecseg8k_local", 
                                            save_dir="/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images")

print("\n✅ Visualizations saved to notebooks/images/")