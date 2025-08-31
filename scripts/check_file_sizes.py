#!/usr/bin/env python3
"""
Check file sizes per requirements.md - max 2000 lines per file
"""
import sys
from pathlib import Path

MAX_LINES = 2000
MAX_LINE_LENGTH = 100

def check_file_size(filepath: Path) -> bool:
    """Check if file meets size requirements"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check total lines
        if len(lines) > MAX_LINES:
            print(f"ERROR: {filepath} has {len(lines)} lines (max: {MAX_LINES})")
            return False
        
        # Check line lengths
        for i, line in enumerate(lines, 1):
            if len(line) > MAX_LINE_LENGTH:
                print(f"ERROR: {filepath}:{i} exceeds {MAX_LINE_LENGTH} characters")
                return False
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to check {filepath}: {e}")
        return False

def main():
    """Check all Python files in src/ directory"""
    src_path = Path('src')
    if not src_path.exists():
        return True  # No src directory yet
    
    all_good = True
    for py_file in src_path.rglob('*.py'):
        if not check_file_size(py_file):
            all_good = False
    
    if not all_good:
        sys.exit(1)
    
    print("All files meet size requirements")

if __name__ == "__main__":
    main()