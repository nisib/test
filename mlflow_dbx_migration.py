# Databricks notebook source
# COMMAND ----------

# DBTITLE 1,MLflow Migration Between Servers
# MAGIC %md
# MAGIC # MLflow Experiment Migration
# MAGIC 
# MAGIC This notebook provides functionality to migrate MLflow experiments between different MLflow tracking servers, 
# MAGIC specifically designed for Databricks environments. This notebook can handle:
# MAGIC 
# MAGIC * Migrating between different Databricks workspaces
# MAGIC * Migrating from open-source MLflow to Databricks
# MAGIC * Migrating from Databricks to open-source MLflow
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC * Access to both source and target MLflow tracking servers
# MAGIC * Appropriate permissions to create experiments and log runs
# MAGIC * If migrating between Databricks workspaces, you'll need authentication tokens for both workspaces

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install mlflow==2.8.0 requests pandas

# COMMAND ----------

# DBTITLE 1,Configuration
# Source and target MLflow tracking URIs
SOURCE_TRACKING_URI = dbutils.widgets.get("source_tracking_uri", "")
TARGET_TRACKING_URI = dbutils.widgets.get("target_tracking_uri", "")

# Experiment selection
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name", "")
EXPERIMENT_ID = dbutils.widgets.get("experiment_id", "")
ALL_EXPERIMENTS = dbutils.widgets.get("all_experiments", "false").lower() == "true"

# Migration options
ONLY_RUNS = dbutils.widgets.get("only_runs", "false").lower() == "true"
OVERWRITE = dbutils.widgets.get("overwrite", "false").lower() == "true"

# Databricks authentication (if needed)
SOURCE_DATABRICKS_TOKEN = dbutils.widgets.get("source_databricks_token", "")
TARGET_DATABRICKS_TOKEN = dbutils.widgets.get("target_databricks_token", "")

# Print configuration (hiding tokens)
print(f"Source tracking URI: {SOURCE_TRACKING_URI}")
print(f"Target tracking URI: {TARGET_TRACKING_URI}")
print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Migrate all experiments: {ALL_EXPERIMENTS}")
print(f"Only migrate runs (no artifacts): {ONLY_RUNS}")
print(f"Overwrite existing experiments: {OVERWRITE}")
print(f"Source Databricks token provided: {'Yes' if SOURCE_DATABRICKS_TOKEN else 'No'}")
print(f"Target Databricks token provided: {'Yes' if TARGET_DATABRICKS_TOKEN else 'No'}")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import tempfile
import shutil
import json
import logging
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

# DBTITLE 1,Helper Functions
def validate_tracking_uri(tracking_uri, token=None):
    """Validate that the tracking URI is accessible"""
    if not tracking_uri:
        raise ValueError("Tracking URI cannot be empty")
    
    parsed_uri = urlparse(tracking_uri)
    
    # Set the token in the environment if provided
    if token and (parsed_uri.scheme == "databricks" or "databricks.com" in parsed_uri.netloc):
        os.environ["DATABRICKS_TOKEN"] = token
    
    # Test connection to the tracking server
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        # Try to list experiments as a connectivity test
        client.list_experiments()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MLflow tracking server at {tracking_uri}: {e}")
        raise ConnectionError(f"Failed to connect to MLflow tracking server at {tracking_uri}: {e}")

def get_experiment_to_migrate(source_client, experiment_name=None, experiment_id=None, all_experiments=False):
    """Get experiment(s) to migrate based on input parameters"""
    if all_experiments:
        experiments = source_client.list_experiments(view_type=ViewType.ALL)
        if not experiments:
            logger.warning("No experiments found on source server")
        return experiments
    
    if experiment_id:
        try:
            experiment = source_client.get_experiment(experiment_id)
            return [experiment]
        except MlflowException as e:
            logger.error(f"Failed to find experiment with ID {experiment_id}: {e}")
            raise ValueError(f"Experiment with ID {experiment_id} not found")
    
    if experiment_name:
        experiment = source_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Failed to find experiment with name {experiment_name}")
            raise ValueError(f"Experiment with name {experiment_name} not found")
        return [experiment]
    
    logger.error("Must provide either experiment_name, experiment_id, or all_experiments=True")
    raise ValueError("Must provide either experiment_name, experiment_id, or all_experiments=True")

def create_experiment_on_target(target_client, experiment_name, existing_experiment_id=None, overwrite=False):
    """Create experiment on target server or use existing one"""
    existing_experiment = target_client.get_experiment_by_name(experiment_name)
    
    if existing_experiment:
        if overwrite:
            logger.info(f"Experiment '{experiment_name}' exists on target server. Using it as target.")
            return existing_experiment.experiment_id
        else:
            logger.warning(f"Experiment '{experiment_name}' already exists on target server.")
            target_experiment_id = existing_experiment.experiment_id
            logger.info(f"Using existing experiment ID: {target_experiment_id}")
            return target_experiment_id
    else:
        try:
            target_experiment_id = target_client.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' on target server with ID: {target_experiment_id}")
            return target_experiment_id
        except Exception as e:
            logger.error(f"Failed to create experiment '{experiment_name}' on target server: {e}")
            raise

def download_artifacts(source_client, run_id, local_dir):
    """Download artifacts from a run to a local directory"""
    try:
        local_path = os.path.join(local_dir, run_id)
        os.makedirs(local_path, exist_ok=True)
        
        # Get list of all artifacts
        artifacts = source_client.list_artifacts(run_id)
        
        if not artifacts:
            logger.info(f"No artifacts found for run {run_id}")
            return local_path
        
        # Download all artifacts
        for artifact in artifacts:
            artifact_path = artifact.path
            if artifact.is_dir:
                nested_artifacts = source_client.list_artifacts(run_id, artifact_path)
                for nested_artifact in nested_artifacts:
                    source_client.download_artifacts(run_id, nested_artifact.path, os.path.join(local_path))
            else:
                source_client.download_artifacts(run_id, artifact_path, os.path.join(local_path))
        
        logger.info(f"Downloaded artifacts for run {run_id} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download artifacts for run {run_id}: {e}")
        raise

def migrate_run(source_client, target_client, run, target_experiment_id, temp_dir, migrate_artifacts=True):
    """Migrate a single run from source to target server"""
    source_run_id = run.info.run_id
    try:
        # Extract run data
        run_data = {
            "tags": {k: v for k, v in run.data.tags.items()},
            "params": {k: v for k, v in run.data.params.items()},
            "metrics": {k: v for k, v in run.data.metrics.items()}
        }
        
        # Handle special tags that might cause issues during migration
        special_tags = ["mlflow.parentRunId", "mlflow.databricks.runURL", "mlflow.user"]
        for tag in special_tags:
            if tag in run_data["tags"]:
                logger.warning(f"Removing '{tag}' tag from run {source_run_id}")
                del run_data["tags"][tag]
        
        # Create the run in the target with start/end times matching source
        start_time = run.info.start_time
        
        # Create the run with the tags from the source run
        new_run = target_client.create_run(
            experiment_id=target_experiment_id,
            start_time=start_time,
            tags=run_data["tags"]
        )
        new_run_id = new_run.info.run_id
        
        # Log parameters
        for key, value in run_data["params"].items():
            target_client.log_param(new_run_id, key, value)
        
        # Log metrics
        for key, value in run_data["metrics"].items():
            target_client.log_metric(new_run_id, key, value)
        
        # Migrate artifacts if requested
        if migrate_artifacts:
            try:
                # Download artifacts to temporary directory
                artifacts_dir = download_artifacts(source_client, source_run_id, temp_dir)
                
                # Log artifacts to the new run
                if os.path.exists(artifacts_dir) and os.listdir(artifacts_dir):
                    target_client.log_artifacts(new_run_id, artifacts_dir)
                    logger.info(f"Migrated artifacts from run {source_run_id} to {new_run_id}")
            except Exception as artifact_e:
                logger.error(f"Failed to migrate artifacts for run {source_run_id}: {artifact_e}")
        
        # Update run status to match the source run
        target_client.set_terminated(new_run_id, run.info.status)
        
        logger.info(f"Successfully migrated run {source_run_id} to {new_run_id}")
        return new_run_id
    except Exception as e:
        logger.error(f"Failed to migrate run {source_run_id}: {e}")
        raise

def migrate_experiment(source_client, target_client, experiment, temp_dir, only_runs=False, overwrite=False):
    """Migrate an entire experiment including all runs and artifacts"""
    experiment_name = experiment.name
    source_experiment_id = experiment.experiment_id
    
    logger.info(f"Migrating experiment '{experiment_name}' (ID: {source_experiment_id})")
    
    # Create experiment on target server
    target_experiment_id = create_experiment_on_target(
        target_client, 
        experiment_name, 
        existing_experiment_id=source_experiment_id,
        overwrite=overwrite
    )
    
    # Get all runs for the experiment
    runs = source_client.search_runs(
        experiment_ids=[source_experiment_id],
        filter_string="",
        run_view_type=ViewType.ALL
    )
    
    logger.info(f"Found {len(runs)} runs to migrate for experiment '{experiment_name}'")
    
    # Migrate each run
    successful_runs = 0
    failed_runs = 0
    failed_run_ids = []
    
    for run in runs:
        try:
            migrate_run(
                source_client, 
                target_client, 
                run, 
                target_experiment_id, 
                temp_dir,
                migrate_artifacts=not only_runs
            )
            successful_runs += 1
        except Exception as e:
            failed_runs += 1
            failed_run_ids.append(run.info.run_id)
            logger.error(f"Failed to migrate run {run.info.run_id}: {e}")
    
    logger.info(f"Experiment migration summary for '{experiment_name}':")
    logger.info(f"  - Total runs: {len(runs)}")
    logger.info(f"  - Successful migrations: {successful_runs}")
    logger.info(f"  - Failed migrations: {failed_runs}")
    
    return {
        "experiment_name": experiment_name,
        "source_experiment_id": source_experiment_id,
        "target_experiment_id": target_experiment_id,
        "total_runs": len(runs),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "failed_run_ids": failed_run_ids
    }

# COMMAND ----------

# DBTITLE 1,Validate Connections
# Validate source and target tracking URIs
logger.info(f"Validating source tracking URI: {SOURCE_TRACKING_URI}")
source_valid = validate_tracking_uri(SOURCE_TRACKING_URI, SOURCE_DATABRICKS_TOKEN)

logger.info(f"Validating target tracking URI: {TARGET_TRACKING_URI}")
target_valid = validate_tracking_uri(TARGET_TRACKING_URI, TARGET_DATABRICKS_TOKEN)

print(f"Source tracking URI validation: {'Successful' if source_valid else 'Failed'}")
print(f"Target tracking URI validation: {'Successful' if target_valid else 'Failed'}")

# Set up clients for source and target servers
source_client = MlflowClient(tracking_uri=SOURCE_TRACKING_URI)
target_client = MlflowClient(tracking_uri=TARGET_TRACKING_URI)

# COMMAND ----------

# DBTITLE 1,List Available Experiments
# List all experiments from source server
source_experiments = source_client.list_experiments()
source_exp_df = pd.DataFrame([{
    "name": exp.name,
    "experiment_id": exp.experiment_id,
    "artifact_location": exp.artifact_location,
    "lifecycle_stage": exp.lifecycle_stage
} for exp in source_experiments])

display(source_exp_df)

# COMMAND ----------

# DBTITLE 1,Perform Migration
# Get experiments to migrate
try:
    experiments = get_experiment_to_migrate(
        source_client,
        experiment_name=EXPERIMENT_NAME if EXPERIMENT_NAME else None,
        experiment_id=EXPERIMENT_ID if EXPERIMENT_ID else None,
        all_experiments=ALL_EXPERIMENTS
    )
    
    logger.info(f"Found {len(experiments)} experiment(s) to migrate")
    
    # Create a temporary directory for artifacts
    temp_dir = f"/dbfs/tmp/mlflow_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temporary directory for artifacts: {temp_dir}")
    
    migration_results = []
    
    # Migrate each experiment
    for experiment in experiments:
        try:
            result = migrate_experiment(
                source_client,
                target_client,
                experiment,
                temp_dir,
                only_runs=ONLY_RUNS,
                overwrite=OVERWRITE
            )
            migration_results.append(result)
        except Exception as e:
            logger.error(f"Failed to migrate experiment {experiment.name}: {e}")
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {e}")
    
    # Print final summary
    logger.info("Migration complete!")
    logger.info(f"Summary: {len(migration_results)} experiment(s) migrated")
    
    # Create a DataFrame with results
    results_df = pd.DataFrame([{
        "experiment_name": result["experiment_name"],
        "source_experiment_id": result["source_experiment_id"],
        "target_experiment_id": result["target_experiment_id"],
        "total_runs": result["total_runs"],
        "successful_runs": result["successful_runs"],
        "failed_runs": result["failed_runs"]
    } for result in migration_results])
    
    display(results_df)
except Exception as e:
    logger.error(f"Migration failed: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Final Results
# Create detailed view of results
if 'migration_results' in locals() and migration_results:
    # Create a list of dictionaries with failed run IDs
    failed_runs_list = []
    for result in migration_results:
        for run_id in result.get("failed_run_ids", []):
            failed_runs_list.append({
                "experiment_name": result["experiment_name"],
                "source_experiment_id": result["source_experiment_id"],
                "failed_run_id": run_id
            })
    
    # Create DataFrame if there are any failed runs
    if failed_runs_list:
        failed_runs_df = pd.DataFrame(failed_runs_list)
        print("Failed runs (if any):")
        display(failed_runs_df)
    else:
        print("All runs migrated successfully!")
else:
    print("No migration results available.")

# COMMAND ----------

# DBTITLE 1,Verify Target Experiments
# List all experiments from target server after migration
target_experiments = target_client.list_experiments()
target_exp_df = pd.DataFrame([{
    "name": exp.name,
    "experiment_id": exp.experiment_id,
    "artifact_location": exp.artifact_location,
    "lifecycle
