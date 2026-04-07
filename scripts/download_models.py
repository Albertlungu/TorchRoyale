import os
import sys

import requests
import roboflow

# Check if API key is available
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("Error: ROBOFLOW_API_KEY environment variable not set")
    print("Please set your API key: export ROBOFLOW_API_KEY=your_key_here")
    sys.exit(1)

try:
    # Initialize Roboflow client
    rf = roboflow.Roboflow(api_key=api_key)

    # Get workspace
    workspace = rf.workspace()
    if workspace is None:
        print("Error: Could not access workspace")
        sys.exit(1)

    print("✓ Workspace access successful")

    # Get project
    project = workspace.project("torchroyale")
    if project is None:
        print("Error: Could not find project 'torchroyale'")
        print("Available projects:")
        try:
            for proj in workspace.list_projects():
                print(f"  - {proj}")
        except:
            print("  (Could not list projects)")
        sys.exit(1)

    print("✓ Project 'torchroyale' found")

    # Get version
    version = project.version(4)
    if version is None:
        print("Error: Could not find version 4 of the project")
        print("Available versions:")
        try:
            for ver in project.versions():
                print(f"  - Version {ver}")
        except:
            print("  (Could not list versions)")
        sys.exit(1)

    print("✓ Version 4 found")

    # Get model
    model = version.model
    if model is None:
        print("Error: Could not access model for version 4")
        sys.exit(1)

    print("✓ Model access successful")

    # Download model
    print("Downloading model...")
    model.download()  # Downloads 'weights.pt' to your local folder
    print("Model downloaded successfully!")

except requests.exceptions.HTTPError as e:
    print(f"HTTP Error during download: {e}")
    if "403" in str(e) or "Forbidden" in str(e):
        print("\nTroubleshooting 403 Forbidden Error:")
        print("1. Your API key can access the project but not download the model")
        print("2. Check if you have download permissions for this specific version")
        print("3. Verify your Roboflow subscription allows model downloads")
        print("4. Some projects may have restricted model downloads")
        print("5. Try downloading from the Roboflow web interface to test permissions")
        print(f"\nAPI Key (first 8 chars): {api_key[:8]}...")
        print("Project URL: https://app.roboflow.com/fafa-zoa5z/torchroyale/4")
    elif "401" in str(e) or "Unauthorized" in str(e):
        print("\nAPI Key appears to be invalid for downloads")
    sys.exit(1)
except roboflow.core.exception.RoboflowException as e:
    print(f"Roboflow API Error: {e}")
    if "403" in str(e) or "Forbidden" in str(e):
        print("\nTroubleshooting 403 Forbidden Error:")
        print("1. Check if your API key has access to the 'torchroyale' project")
        print("2. Verify the project exists in your workspace")
        print("3. Ensure version 4 exists and is accessible")
        print("4. Check if you have download permissions for this project")
        print("5. Verify your Roboflow subscription allows model downloads")
        print(f"\nAPI Key (first 8 chars): {api_key[:8]}...")
    elif "401" in str(e) or "Unauthorized" in str(e):
        print("\nAPI Key appears to be invalid or expired")
        print("Please check your ROBOFLOW_API_KEY environment variable")
except Exception as e:
    print(f"Unexpected error: {e}")
    print(f"Error type: {type(e).__name__}")

    # Additional troubleshooting for download issues
    if "403" in str(e) or "Forbidden" in str(e):
        print("\nThis appears to be a permissions issue during download.")
        print("The project is accessible but the model download is restricted.")
        print(
            "Please check your Roboflow account permissions or contact the project owner."
        )

    sys.exit(1)
