import os

def final_verify_mlflow(root_dir):
    all_good = True
    for root, dirs, files in os.walk(root_dir):
        if 'meta.yaml' in files:
            file_path = os.path.join(root, 'meta.yaml')
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common corruption/wrong path markers
            if len(content.strip()) == 0:
                print(f"❌ EMPTY FILE: {file_path}")
                all_good = False
            elif "artifact_uri:" in content and "file:///" not in content:
                print(f"❌ MISSING file:/// in artifact_uri: {file_path}")
                all_good = False
            elif "artifact_location:" in content and "file:///" not in content:
                print(f"❌ MISSING file:/// in artifact_location: {file_path}")
                all_good = False
            elif "/content/mlruns" in content:
                print(f"❌ UNCONVERTED Colab path: {file_path}")
                all_good = False
            elif "\\" in content and ("artifact_uri" in content or "artifact_location" in content):
                # We check for backslashes in the URI lines
                for line in content.splitlines():
                    if ("artifact_uri" in line or "artifact_location" in line) and "\\" in line:
                        print(f"❌ BACKSLASH in path: {file_path} -> {line}")
                        all_good = False
            
    if all_good:
        print("ALL meta.yaml files are correctly formatted!")
    else:
        print("⚠️ Some files still have issues.")

if __name__ == "__main__":
    final_verify_mlflow('mlruns')
