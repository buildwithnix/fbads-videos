#!/usr/bin/env python3
"""
Final fix - add Ray decorators properly
"""

# Read the file
with open('cluster_processor.py', 'r') as f:
    lines = f.readlines()

# Fix the file line by line
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # If we find a class definition that needs @ray.remote
    if line.strip().startswith('class ModelActor') or line.strip().startswith('class CacheActor'):
        # Add @ray.remote before the class
        fixed_lines.append('@ray.remote\n')
        fixed_lines.append(line)
    # Fix async def to def
    elif 'async def' in line:
        fixed_lines.append(line.replace('async def', 'def'))
    # Remove await
    elif 'await ' in line:
        fixed_lines.append(line.replace('await ', ''))
    # Fix the main function call
    elif 'asyncio.run(main())' in line:
        fixed_lines.append('    main()\n')
    else:
        fixed_lines.append(line)
    
    i += 1

# Write the fixed version
with open('cluster_processor_final.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Created cluster_processor_final.py with all fixes!")
print("This should work!")