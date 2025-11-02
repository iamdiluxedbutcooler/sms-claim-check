import re
import sys
from pathlib import Path

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\u2600-\u26FF\u2700-\u27BF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_docstring = False
    docstring_char = None
    
    for line in lines:
        stripped = line.strip()
        
        if '"""' in stripped or "'''" in stripped:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in stripped else "'''"
                continue
            elif docstring_char in stripped:
                in_docstring = False
                continue
        
        if in_docstring:
            continue
        
        if stripped.startswith('#'):
            continue
        
        line = remove_emojis(line)
        new_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Processed: {filepath}")

def main():
    src_dir = Path("src")
    
    for pyfile in src_dir.rglob("*.py"):
        process_file(pyfile)
    
    for pyfile in Path(".").glob("*.py"):
        if pyfile.name not in ["remove_comments.py"]:
            process_file(pyfile)
    
    print("Done!")

if __name__ == "__main__":
    main()
