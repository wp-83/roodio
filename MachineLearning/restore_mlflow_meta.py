import os
import time

def restore_meta_yaml():
    base_dir = r"C:\CAWU4GROUP3\projects\projectRoodio\machineLearning\mlruns"
    exp_id = "274189782535581600"
    exp_name = "Hierarchical_Stage2B_Hybrid"
    
    # 1. Restore Experiment meta.yaml
    exp_path = os.path.join(base_dir, exp_id)
    exp_meta = os.path.join(exp_path, "meta.yaml")
    
    exp_content = f"""artifact_location: file:///{base_dir.replace('\\', '/')}/{exp_id}
creation_time: {int(time.time() * 1000)}
experiment_id: '{exp_id}'
last_update_time: {int(time.time() * 1000)}
lifecycle_stage: active
name: {exp_name}
"""
    with open(exp_meta, 'w', encoding='utf-8') as f:
        f.write(exp_content)
    print(f"Restored experiment meta: {exp_meta}")

    # 2. Restore Runs
    runs = [
        "0f14f57c67fb4a919bc1cd9ce0bd5d7d",
        "64b1ac12e87c4115b92089796ee36fea",
        "9960c3ebfafa4c439218ee23bcd1062a"
    ]
    
    for run_id in runs:
        run_path = os.path.join(exp_path, run_id)
        run_meta = os.path.join(run_path, "meta.yaml")
        
        # Try to get run name from tags if available
        run_name = run_id
        name_tag_path = os.path.join(run_path, "tags", "mlflow.runName")
        if os.path.exists(name_tag_path):
            with open(name_tag_path, 'r') as f:
                run_name = f.read().strip()

        run_content = f"""artifact_uri: file:///{base_dir.replace('\\', '/')}/{exp_id}/{run_id}/artifacts
end_time: {int(time.time() * 1000)}
entry_point_name: ''
experiment_id: '{exp_id}'
lifecycle_stage: active
run_id: {run_id}
run_name: {run_name}
source_name: ''
source_type: 4
source_version: ''
start_time: {int(time.time() * 1000) - 300000}
status: 3
tags: []
user_id: unknown
"""
        with open(run_meta, 'w', encoding='utf-8') as f:
            f.write(run_content)
        print(f"Restored run meta: {run_meta}")

if __name__ == "__main__":
    restore_meta_yaml()
