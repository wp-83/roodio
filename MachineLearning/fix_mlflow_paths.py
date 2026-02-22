import os
import re

def fix_mlflow_paths(root_dir):
    root_abs_path = os.path.abspath(root_dir).replace('\\', '/')
    if not root_abs_path.startswith('/'):
        root_abs_path = '/' + root_abs_path
    
    base_uri = f"file://{root_abs_path}"
    print(f"Base URI for replacement: {base_uri}")

    count = 0
    for root, dirs, files in os.walk(root_dir):
        if 'meta.yaml' in files:
            file_path = os.path.join(root, 'meta.yaml')
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. Fix Linux/Colab paths
            new_content = content.replace('/content/mlruns', base_uri)
            
            # 2. Fix raw Windows paths (missing file:///)
            # Look for lines like: artifact_uri: C:\path... or artifact_location: C:\path...
            def windows_to_uri(match):
                path = match.group(2).replace('\\', '/')
                if not path.startswith('/'):
                    path = '/' + path
                return f"{match.group(1)} file://{path}"

            new_content = re.sub(r'(artifact_uri|artifact_location): ([A-Z]:[^\n]+)', windows_to_uri, new_content)

            if content != new_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed: {file_path}")
                count += 1
    
    print(f"\nTotal meta.yaml files fixed: {count}")

if __name__ == "__main__":
    fix_mlflow_paths('mlruns')
