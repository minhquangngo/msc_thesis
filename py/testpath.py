import os
experiment = 208039388113350502
run = "0c861f5f9a874e05b04e43bb6341bd96"

# Check the actual path
path = f"py/mlartifacts/{experiment}/{run}"
print(f"Checking: {path}")
print(f"Exists: {os.path.exists(path)}")

if os.path.exists(path):
    for root, dirs, files in os.walk(path):
        print(f"Directory: {root}")
        print(f"Files: {files}")
        print("---")