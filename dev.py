#!/usr/bin/env python3
import os
import signal
import subprocess
import sys

def kill_streamlit_processes():
    """Kill existing Streamlit processes if any are running."""
    try:
        print("Killing existing Streamlit process:", end=" ")
        result = subprocess.run(
            ["pkill", "-f", "streamlit run app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            print("Success!")
        else:
            print("No processes found.")
    except Exception as e:
        print(f"Error killing processes: {e}")

def main():
    """Run the Streamlit app with development settings."""
    print("Starting Crawl4AI development server with hot reload...")
    
    # Kill any existing Streamlit processes
    kill_streamlit_processes()
    
    # Ensure the .streamlit directory exists
    os.makedirs(".streamlit", exist_ok=True)
    
    # Create or update the config.toml file with hot reload settings
    config_content = """
[server]
# Enable hot reload
runOnSave = true
headless = true
fileWatcherType = "auto"

[browser]
serverAddress = "localhost"
gatherUsageStats = false
serverPort = 8501

[theme]
base = "light"
"""
    
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content.strip())
    
    # Run Streamlit with explicit hot reload flag
    cmd = ["streamlit", "run", "app.py", "--server.runOnSave=true"]
    
    try:
        # Start the process and make it the process group leader
        process = subprocess.Popen(
            cmd,
            start_new_session=True,
        )
        
        # Wait for the process to complete (or Ctrl+C)
        process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nStopping development server...")
        
        try:
            # Send signal to process group to terminate all child processes
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
        
        sys.exit(0)

if __name__ == "__main__":
    main() 