"""I/O utilities for JSON file handling and hashing."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file from path.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON content as dictionary
    """
    return json.loads(Path(path).read_text())


def save_json(path: Union[str, Path], obj: Dict[str, Any], indent: int = 2) -> None:
    """Save dictionary to JSON file.
    
    Args:
        path: Path to save JSON file
        obj: Dictionary to save
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent))


def config_hash(obj: Dict[str, Any]) -> str:
    """Generate deterministic hash for configuration object.
    
    Args:
        obj: Configuration dictionary
        
    Returns:
        12-character hash string
    """
    canon = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:12]


def file_hash(path: Union[str, Path]) -> str:
    """Generate SHA256 hash of file contents.
    
    Args:
        path: Path to file
        
    Returns:
        Full SHA256 hash string
    """
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path