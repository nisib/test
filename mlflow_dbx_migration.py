

# Databricks notebook source
# COMMAND ----------

# DBTITLE 1,Run PySpark Code from S3
# Configure access to S3
spark.conf.set("spark.hadoop.fs.s3a.access.key", "YOUR_AWS_ACCESS_KEY")
spark.conf.set("spark.hadoop.fs.s3a.secret.key", "YOUR_AWS_SECRET_KEY")
spark.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

# COMMAND ----------

# DBTITLE 1,Define S3 paths
S3_BUCKET = "your-s3-bucket"
S3_PROJECT_PATH = "projects/pyspark_analytics"
MAIN_SCRIPT = "main.py"

# Full path to the main script
s3_main_script_path = f"s3a://{S3_BUCKET}/{S3_PROJECT_PATH}/{MAIN_SCRIPT}"

# Display the path
print(f"Main script path: {s3_main_script_path}")

# COMMAND ----------

# DBTITLE 1,List files in the project directory
# List files in the S3 project directory
files = dbutils.fs.ls(f"s3a://{S3_BUCKET}/{S3_PROJECT_PATH}")
display(files)

# COMMAND ----------

# DBTITLE 1,Method 1: Import and Run Main Function
# This method works if your main.py has a main() function that you want to run

# Import the main module dynamically
import importlib.util
import sys
from pyspark.dbutils import DBUtils

# Copy the main script to a temporary location where Python can import it
temp_dir = "/dbfs/tmp/s3_import"
dbutils.fs.mkdirs(temp_dir)
temp_script_path = f"{temp_dir}/main.py"

# Copy from S3 to local filesystem
dbutils.fs.cp(s3_main_script_path, f"file:{temp_script_path}")

# Import the module
spec = importlib.util.spec_from_file_location("s3_main", temp_script_path)
s3_main = importlib.util.module_from_spec(spec)
sys.modules["s3_main"] = s3_main
spec.loader.exec_module(s3_main)

# Run the main function
print("Running main() function from imported module...")
s3_main.main()

# COMMAND ----------

# DBTITLE 1,Method 2: Execute Script Directly
# This method runs the script as a standalone program

# Import dbutils
from pyspark.dbutils import DBUtils

# Execute the script from S3
print(f"Executing script: {s3_main_script_path}")
dbutils.py.run(s3_main_script_path)

# COMMAND ----------

# DBTITLE 1,Method 3: Spark Submit
# This method uses spark-submit equivalent to run the script

# Submit the job to Spark
script_path = f"/dbfs/mnt/s3-mount/{S3_BUCKET}/{S3_PROJECT_PATH}/{MAIN_SCRIPT}"

# Import required libraries
import subprocess
import os

# Create a temporary script to run spark-submit
submit_script = """
#!/bin/bash
spark-submit \
  --master local[*] \
  --conf spark.hadoop.fs.s3a.access.key=YOUR_AWS_ACCESS_KEY \
  --conf spark.hadoop.fs.s3a.secret.key=YOUR_AWS_SECRET_KEY \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  {script_path}
""".format(script_path=script_path)

# Write the script to a file
with open("/tmp/submit_script.sh", "w") as f:
    f.write(submit_script)

# Make it executable
os.chmod("/tmp/submit_script.sh", 0o755)

# Run the script
print("Running spark-submit script...")
subprocess.call(["/tmp/submit_script.sh"])

# COMMAND ----------

# DBTITLE 1,Method 4: Run via %run Magic Command
# This method works if you've mounted the S3 bucket to DBFS

# Mount S3 bucket if not already mounted
mount_point = "/mnt/s3-mount"
if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(f"s3a://{S3_BUCKET}", mount_point)

# Path to the script in the mounted directory
mounted_script_path = f"{mount_point}/{S3_PROJECT_PATH}/{MAIN_SCRIPT}"

# Run the script
print(f"Running script from mounted path: {mounted_script_path}")
%run $mounted_script_path

# COMMAND ----------

# DBTITLE 1,Cleanup
# Unmount if needed
if any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.unmount(mount_point)

# Remove temporary files
dbutils.fs.rm(temp_dir, recurse=True)
