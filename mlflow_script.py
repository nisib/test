#!/usr/bin/env python
# mlflow_migration.py - Script to migrate MLflow experiments between servers
import os
import tempfile
import shutil
import json
import argparse
import logging
from urllib.parse import urlparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argument_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description="Migrate MLflow experiments between servers")
    parser.add_argument("--source-tracking-uri", required=True, help="Source MLflow tracking server URI")
    parser.add_argument("--target-tracking-uri", required=True, help="Target MLflow tracking server URI")
    parser.add_argument("--experiment-name", help="Name of the specific experiment to migrate (optional)")
    parser.add_argument("--experiment-id", help="ID of the specific experiment to migrate (optional)")
    parser.add_argument("--all-experiments", action="store_true", help="Migrate all experiments")
    parser.add_argument("--only-runs", action="store_true", help="Only migrate runs, not artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment on target server")
    return parser

def validate_tracking_uri(tracking_uri):
    """Validate that the tracking URI is accessible"""
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme not in ["http", "https", "file", "postgresql", "mysql", "sqlite"]:
        logger.error(f"Unsupported tracking URI scheme: {parsed_uri.scheme}")
        raise ValueError(f"Unsupported tracking URI scheme: {parsed_uri.scheme}")
    
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
    
    logger.error("Must provide either --experiment-name, --experiment-id, or --all-experiments")
    raise ValueError("Must provide either --experiment-name, --experiment-id, or --all-experiments")

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
        artifact_uri = source_client.get_run(run_id).info.artifact_uri
        local_path = os.path.join(local_dir, run_id)
        os.makedirs(local_path, exist_ok=True)
        
        # Get list of all artifacts
        artifacts = source_client.list_artifacts(run_id)
        
        if not artifacts:
            logger.info(f"No artifacts found for run {run_id}")
            return local_path
        
        # Download all artifacts
        for artifact in artifacts:
            if artifact.is_dir:
                # Recursively download directory
                source_client.download_artifacts(run_id, artifact.path, local_path)
            else:
                # Download individual file
                source_client.download_artifacts(run_id, artifact.path, os.path.join(local_path, os.path.dirname(artifact.path)))
        
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
        if "mlflow.parentRunId" in run_data["tags"]:
            logger.warning(f"Removing 'mlflow.parentRunId' tag from run {source_run_id}")
            del run_data["tags"]["mlflow.parentRunId"]
        
        # Create the run in the target
        new_run = target_client.create_run(
            experiment_id=target_experiment_id,
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
            # Download artifacts to temporary directory
            artifacts_dir = download_artifacts(source_client, source_run_id, temp_dir)
            
            # Log artifacts to the new run
            if os.path.exists(artifacts_dir) and os.listdir(artifacts_dir):
                target_client.log_artifacts(new_run_id, artifacts_dir)
                logger.info(f"Migrated artifacts from run {source_run_id} to {new_run_id}")
        
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
        "failed_runs": failed_runs
    }

def main():
    """Main function to migrate MLflow experiments"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate source and target tracking URIs
    logger.info(f"Validating source tracking URI: {args.source_tracking_uri}")
    validate_tracking_uri(args.source_tracking_uri)
    
    logger.info(f"Validating target tracking URI: {args.target_tracking_uri}")
    validate_tracking_uri(args.target_tracking_uri)
    
    # Set up clients for source and target servers
    source_client = MlflowClient(tracking_uri=args.source_tracking_uri)
    target_client = MlflowClient(tracking_uri=args.target_tracking_uri)
    
    # Get experiments to migrate
    experiments = get_experiment_to_migrate(
        source_client,
        experiment_name=args.experiment_name,
        experiment_id=args.experiment_id,
        all_experiments=args.all_experiments
    )
    
    logger.info(f"Found {len(experiments)} experiment(s) to migrate")
    
    # Create a temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
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
                    only_runs=args.only_runs,
                    overwrite=args.overwrite
                )
                migration_results.append(result)
            except Exception as e:
                logger.error(f"Failed to migrate experiment {experiment.name}: {e}")
    
    # Print final summary
    logger.info("Migration complete!")
    logger.info(f"Summary: {len(migration_results)} experiment(s) migrated")
    
    for result in migration_results:
        logger.info(f"Experiment '{result['experiment_name']}':")
        logger.info(f"  - Source ID: {result['source_experiment_id']}")
        logger.info(f"  - Target ID: {result['target_experiment_id']}")
        logger.info(f"  - Runs: {result['successful_runs']}/{result['total_runs']} successful")

if __name__ == "__main__":
    main()
