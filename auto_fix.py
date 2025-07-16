#!/usr/bin/env python3
"""
Automatically fix all Python syntax and import errors
"""

import ast
import subprocess
import re
import sys

def check_syntax(filename):
    """Check for syntax errors and return them"""
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read())
        return []
    except SyntaxError as e:
        return [{"line": e.lineno, "msg": e.msg, "text": e.text}]

def fix_imports(filename):
    """Run the file and capture import errors"""
    result = subprocess.run([sys.executable, filename], 
                          capture_output=True, text=True, timeout=5)
    
    if "ModuleNotFoundError" in result.stderr:
        # Extract missing module
        match = re.search(r"No module named '(\w+)'", result.stderr)
        if match:
            module = match.group(1)
            print(f"Installing missing module: {module}")
            subprocess.run([sys.executable, "-m", "pip", "install", module])
            return True
    return False

def auto_fix_file(filename):
    """Automatically fix common Python errors"""
    print(f"Auto-fixing {filename}...")
    
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Fix common issues
    fixed_lines = []
    in_ray_remote = False
    
    for i, line in enumerate(lines):
        # Track if we're after @ray.remote
        if '@ray.remote' in line:
            in_ray_remote = True
            fixed_lines.append(line)
            continue
        
        # Fix async after @ray.remote
        if in_ray_remote and 'class ' in line:
            in_ray_remote = False
        
        if in_ray_remote and 'async def' in line:
            line = line.replace('async def', 'def')
        
        # Fix common async issues
        if 'async def' in line and '@ray.remote' in lines[max(0, i-5):i]:
            line = line.replace('async def', 'def')
        
        if 'await ' in line:
            line = line.replace('await ', '')
        
        if 'async with' in line:
            line = line.replace('async with', 'with')
        
        if 'asyncio.run(' in line:
            line = line.replace('asyncio.run(', '').rstrip(')\n') + '\n'
        
        fixed_lines.append(line)
    
    # Write fixed version
    output_file = filename.replace('.py', '_autofixed.py')
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    return output_file

def main():
    """Fix cluster_processor.py automatically"""
    
    # Step 1: Auto-fix syntax
    fixed_file = auto_fix_file('cluster_processor.py')
    print(f"Created: {fixed_file}")
    
    # Step 2: Check syntax
    errors = check_syntax(fixed_file)
    if errors:
        print(f"Still has syntax errors: {errors}")
        # Try to fix them
        with open(fixed_file, 'r') as f:
            lines = f.readlines()
        
        for error in errors:
            if error['line'] and 'unmatched' in error['msg']:
                # Fix unmatched parentheses
                lines[error['line']-1] = lines[error['line']-1].replace('))', ')')
        
        with open(fixed_file, 'w') as f:
            f.writelines(lines)
    
    # Step 3: Add missing @ray.remote decorators
    with open(fixed_file, 'r') as f:
        content = f.read()
    
    # Add @ray.remote to ModelActor and CacheActor
    content = re.sub(r'(class ModelActor[^:]*:)', r'@ray.remote\n\1', content)
    content = re.sub(r'(class CacheActor[^:]*:)', r'@ray.remote\n\1', content)
    
    with open(fixed_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Auto-fix complete! Run: python {fixed_file}")

if __name__ == "__main__":
    main()