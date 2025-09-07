# Package Requirements and Dependencies

## LaTeX Packages

### Required Packages
These packages must be included in your LaTeX preamble:

```latex
\usepackage{graphicx}    % For including graphics
\usepackage{booktabs}    % For professional tables
\usepackage{amsmath}     % For mathematical expressions
\usepackage{array}       % For advanced table formatting
\usepackage{multirow}    % For multi-row cells in tables
\usepackage{caption}     % For caption formatting
\usepackage{geometry}    % For page margins
\usepackage{hyperref}    % For hyperlinks (load last)
```

### Optional But Useful
```latex
\usepackage{xcolor}      % For colored text
\usepackage{listings}    % For code listings
\usepackage{enumitem}    % For customized lists
```

### Packages to AVOID

**Do NOT use these packages** as they cause compatibility issues:

```latex
% AVOID - Not compatible with many journal classes
\usepackage{subfigure}   % Deprecated, causes issues
\usepackage{subcaption}  % Not supported by JMLR and others
\usepackage{tablenotes}  % Requires threeparttable
\usepackage{threeparttable} % Not always available

% Use alternatives shown in latex_formatting.md
```

## Python Dependencies

### Core Requirements
```txt
# requirements.txt
matplotlib>=3.5.0     # For figure generation
numpy>=1.20.0        # For numerical operations
pandas>=1.3.0        # For data manipulation
```

### Installation
```bash
pip install matplotlib numpy pandas
```

### Matplotlib Backend
For server environments without display:
```python
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
```

## Font Configuration

### System Fonts
Ensure these fonts are available:
- DejaVu Sans (default fallback)
- Helvetica (professional look)
- Arial (Windows compatibility)

### Matplotlib Font Setup
```python
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
```

### LaTeX Font in Matplotlib (Optional)
For LaTeX-rendered text in figures:
```python
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
```

## Project Structure

### Expected Directory Layout
```
project_root/
├── src/
│   └── endopoint/
│       └── vis/
│           ├── __init__.py
│           ├── latex_tables.py
│           ├── bar_charts.py
│           └── generate_paper_figures.py
├── results/
│   └── pointing_original/
│       ├── zero_shot/
│       ├── fewshot_standard/
│       └── fewshot_hard_negatives/
├── guidelines/
│   ├── latex_formatting.md
│   ├── figure_generation.md
│   └── package_requirements.md
└── paper/
    ├── figures/     # LaTeX table files
    └── images/      # PDF figure files
```

## Import Statements

### Python Imports
Standard import order for visualization scripts:

```python
#!/usr/bin/env python
"""Module docstring"""

# Standard library
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Before pyplot import
import matplotlib.pyplot as plt

# Local imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from endopoint.vis.latex_tables import generate_metrics_latex_table
from endopoint.vis.bar_charts import generate_model_performance_bars
```

### LaTeX Document Preamble
Minimal working preamble:

```latex
\documentclass[11pt]{article}  % or {jmlr} for conference

% Essential packages
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{array}

% Page setup
\usepackage[margin=1in]{geometry}

% Hyperlinks (load last)
\usepackage{hyperref}

\begin{document}
% Content here
\end{document}
```

## Environment Setup

### Conda Environment
```yaml
# environment.yml
name: llm_cholec_organ
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - matplotlib=3.5
  - numpy=1.20
  - pandas=1.3
  - pip
  - pip:
    - -e .  # Install local package
```

### Docker Setup
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Package subfigure not found"
**Solution**: Remove subfigure, use tabular environment instead

#### Issue: "Font DejaVu Sans not found"
**Solution**: Install system fonts
```bash
# Ubuntu/Debian
sudo apt-get install fonts-dejavu-core

# macOS
brew install --cask font-dejavu

# Or use matplotlib's default
plt.rcParams['font.family'] = 'sans-serif'
```

#### Issue: "Cannot import matplotlib.pyplot"
**Solution**: Set backend before import
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

#### Issue: "LaTeX Error: tablenotes undefined"
**Solution**: Use parbox instead (see latex_formatting.md)

## Version Compatibility

### Tested Configurations
- Python: 3.8, 3.9, 3.10, 3.11
- Matplotlib: 3.5+
- NumPy: 1.20+
- LaTeX: TeX Live 2020+

### Journal-Specific Requirements
- **JMLR**: No subfigure/subcaption support
- **NeurIPS**: Requires specific style file
- **CVPR**: IEEE format requirements
- **MICCAI**: Springer LNCS format

Always check conference/journal guidelines for specific package restrictions.