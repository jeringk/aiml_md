import os
import re
import urllib.parse

base_dir = '/Users/jingo/Library/CloudStorage/GoogleDrive-jeringeok@gmail.com/My Drive/Personal/Learning/AIML/AIMLBits/Sem3/MarkDown/SMA'

renames = {}

# Gather renames for all files
for root, dirs, files in os.walk(base_dir):
    if '.git' in root or '.agent' in root:
        continue
    for fname in files:
        if not fname.endswith('.md'):
            continue
        
        fpath = os.path.join(root, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        heading = None
        heading_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('# '):
                heading = line[2:].strip()
                heading_idx = i
                break
                
        if heading:
            # Sanitize heading
            safe_heading = heading.replace(':', ' -').replace('/', '-').replace('?', '').replace('\\', '-')
            new_fname = f"{safe_heading}.md"
            
            if new_fname != fname:
                start_idx = heading_idx + 1
                while start_idx < len(lines) and lines[start_idx].strip() == '':
                    start_idx += 1
                new_lines = lines[start_idx:]
                
                # Check for uniqueness to avoid overwriting
                if new_fname in [d['new_fname'] for d in renames.values()]:
                    print(f"Warning: collision for {new_fname}")
                
                renames[fpath] = {
                    'old_fname': fname,
                    'new_fname': new_fname, 
                    'new_content': "".join(new_lines),
                    'safe_heading': safe_heading,
                    'root': root
                }

print(f"Found {len(renames)} files to rename:")
for old_fpath, data in renames.items():
    print(f"  {data['old_fname']} -> {data['new_fname']}")

