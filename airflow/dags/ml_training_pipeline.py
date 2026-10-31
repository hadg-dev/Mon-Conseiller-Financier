import boto3
import time
import requests
import base64
import os
import paramiko
import re
import shlex
from datetime import datetime, timedelta
from airflow.sdk import task
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.ec2 import (
    EC2CreateInstanceOperator,
    EC2TerminateInstanceOperator,
)
from airflow.task.trigger_rule import TriggerRule

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# GitHub Config
GITHUB_REPO = os.getenv("GITHUB_REPO") 
# User changed env var name to GITHUB_PAT, updating here:
GITHUB_TOKEN = os.getenv("GITHUB_PAT") 
BRANCH_NAME = os.getenv("BRANCH_NAME", "main")

# Docker Config
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE")

# AWS Config
KEY_PAIR_NAME = os.getenv("KEY_PAIR_NAME") 
AMI_ID = os.getenv("AMI_ID", "ami-00ac45f3035ff009e") 
SECURITY_GROUP_ID = os.getenv("SECURITY_GROUP_ID")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "t3.small")

# Connection (Direct Env Vars)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("AWS_DEFAULT_REGION", "eu-west-3")
ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI")

# MLFlow Config
REPORT_TRAINING = os.getenv("REPORT_TRAINING", "mlflow")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mon_conseiller_financier")
# Only needed when the MLflow Space is private
MLFLOW_TRACKING_TOKEN = os.getenv("MLFLOW_TRACKING_TOKEN")

# overrides
NB_EPOCHS = os.getenv("NB_EPOCHS", 1)
STEPS = os.getenv("STEPS", "train,score_perplexity,copy_s3")
# Tiny model to test the pipeline on a small instance; real model: Qwen/Qwen3.5-4B
MODEL_NAME = os.getenv("MODEL_NAME", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

# User Data (Install Docker)
USER_DATA_SCRIPT = """#!/bin/bash
# 1. Redirect all output to a dedicated log file for easy debugging
exec > /var/log/user-data-debug.log 2>&1
set -x # Print every command before executing

echo "--> STARTING SETUP"

# 2. Prevent apt-get from asking questions
export DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies
echo "--> Installing Apt Packages..."
apt-get update
apt-get install -y docker.io

# 4. Configure Docker
echo "--> Configuring Docker..."
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu
# Ensure socket is ready before chmod
while [ ! -S /var/run/docker.sock ]; do echo "Waiting for Docker Socket..."; sleep 1; done
chmod 666 /var/run/docker.sock

# 5. Add swap, small instances run out of memory while loading the libraries
echo "--> Adding Swap..."
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# 6. Signal Success
echo "--> SETUP COMPLETE"
touch /tmp/airflow_ready
"""

DAG_ID = 'github_ec2_ml_training'
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id=DAG_ID,
    schedule=None,  # <--- FIX: Changed from schedule_interval to schedule
    default_args=default_args,
    catchup=False,
    tags=['github', 'ec2', 'ml-training'],
) as dag:

    # ------------------------------------------------------------------
    # TASK 1: Poll GitHub Actions Status
    # ------------------------------------------------------------------
    @task
    def wait_for_github_ci():
        if not GITHUB_TOKEN or not GITHUB_REPO:
            raise ValueError("Missing GITHUB_PAT or GITHUB_REPO env vars")

        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        params = {"branch": BRANCH_NAME, "status": "completed", "per_page": 1}

        print(f"📡 Polling GitHub Actions for {GITHUB_REPO}...")

        # Poll for 10 minutes (20 checks * 30 seconds)
        for _ in range(20): 
            try:
                response = requests.get(api_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                if not data['workflow_runs']:
                    print("No workflow runs found yet.")
                    time.sleep(30)
                    continue

                latest_run = data['workflow_runs'][0]
                conclusion = latest_run['conclusion']
                
                print(f"🔎 Latest Run Conclusion: {conclusion}")

                if conclusion == 'success':
                    print("✅ CI Passed!")
                    return True
                elif conclusion == 'failure':
                    raise Exception("❌ Latest CI run failed!")
                
            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(30)

        raise TimeoutError("Timed out waiting for GitHub CI.")

    # ------------------------------------------------------------------
    # TASK 2: Create EC2 Instance
    # ------------------------------------------------------------------
    create_ec2 = EC2CreateInstanceOperator(
        task_id="create_ec2_instance",
        region_name=REGION_NAME, 
        image_id=AMI_ID,
        max_count=1,
        min_count=1,
        config={
            "InstanceType": INSTANCE_TYPE,
            "KeyName": KEY_PAIR_NAME,
            "SecurityGroupIds": [SECURITY_GROUP_ID] if SECURITY_GROUP_ID else [],
            "UserData": base64.b64encode(USER_DATA_SCRIPT.encode("ascii")).decode("ascii"),
            # The default 8 GB root disk cannot hold the training image
            "BlockDeviceMappings": [{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': 30, 'VolumeType': 'gp3', 'DeleteOnTermination': True}}],
            "TagSpecifications": [{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Airflow-Trainer'}]}]
        },
        wait_for_completion=True
    )

    # ------------------------------------------------------------------
    # TASK 3: Check EC2 Status Checks (2/2)
    # ------------------------------------------------------------------
    @task
    def check_ec2_status(instance_ids):
        instance_id = instance_ids[0]
        # Boto3 client auto-reads credentials from os.environ
        client = boto3.client('ec2', region_name=REGION_NAME)
        
        print(f"⏳ Waiting for Instance {instance_id} to pass status checks...")
        waiter = client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[instance_id])
        
        print(f"✅ Instance {instance_id} is ready.")
        return instance_id

    # ------------------------------------------------------------------
    # TASK 4: Get Public IP
    # ------------------------------------------------------------------
    @task
    def get_public_ip(instance_id):
        ec2 = boto3.resource('ec2', region_name=REGION_NAME)
        instance = ec2.Instance(instance_id)
        return instance.public_ip_address

    # ------------------------------------------------------------------
    # TASK 5: Run Training (SSH with mounted key)
    # ------------------------------------------------------------------
    @task
    def run_training_via_ssh(public_ip, dag_run=None):
        print(f"🔌 Connecting to {public_ip}...")

        if not DOCKER_IMAGE:
            raise ValueError("DOCKER_IMAGE is missing!")

        # GitHub Actions sends the commit hash; manual runs use the latest image
        image_tag = (dag_run.conf or {}).get("git_hash", "latest")
        # The tag comes from the API and ends up in a remote shell command
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", str(image_tag)):
            raise ValueError(f"Invalid git_hash: {image_tag!r}")
        image = f"{DOCKER_IMAGE}:{image_tag}"

        ssh_key = paramiko.RSAKey.from_private_key_file(
            "/opt/airflow/airflow-training-key.pem"
        )

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # 1. Connect
            for i in range(10):
                try:
                    ssh.connect(public_ip, username='ubuntu', pkey=ssh_key, timeout=10)
                    break
                except Exception as e:
                    print(f"Waiting for SSH... ({e})")
                    time.sleep(10)

            # 2. Wait for the User Data script to finish installing Docker
            print("⏳ Waiting for Docker installation to finish...")

            wait_cmd = """
            for i in {1..60}; do
                if [ -f /tmp/airflow_ready ]; then
                    echo "✅ Docker is ready!"
                    exit 0
                fi
                echo "Waiting for setup..."
                sleep 5
            done
            echo "❌ Timed out waiting for setup"
            exit 1
            """
            
            stdin, stdout, stderr = ssh.exec_command(wait_cmd)
            if stdout.channel.recv_exit_status() != 0:
                print(stderr.read().decode())
                raise Exception("Environment Setup Failed")
                
            # ============================================================
            # 🔍 DEBUG STEP: DUMP STARTUP LOGS
            # ============================================================
            # DEBUG STEP: Read our custom log file
            print("\n🔍 READING USER DATA DEBUG LOG:")
            # Note: checking /var/log/user-data-debug.log instead of cloud-init-output.log
            stdin, stdout, stderr = ssh.exec_command("cat /var/log/user-data-debug.log")
            print(stdout.read().decode())

            # 3. Run Training
            container_env = {
                "REPORT_TRAINING": REPORT_TRAINING,
                "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
                # settings.py reads the experiment name from MLFLOW_EXPERIMENT_ID
                "MLFLOW_EXPERIMENT_ID": MLFLOW_EXPERIMENT_NAME,
                "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
                "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
                "AWS_DEFAULT_REGION": REGION_NAME,
                "ARTIFACT_STORE_URI": ARTIFACT_STORE_URI,
                "NB_EPOCHS": NB_EPOCHS,
                "STEPS": STEPS,
                "MODEL_NAME": MODEL_NAME,
            }
            if MLFLOW_TRACKING_TOKEN:
                container_env["MLFLOW_TRACKING_TOKEN"] = MLFLOW_TRACKING_TOKEN

            exports = "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in container_env.items())
            env_flags = " ".join(f"-e {name}" for name in container_env)

            cmd = f"""
set -e
{exports}

echo {shlex.quote(f"📥 Pulling {image}...")}
docker pull {shlex.quote(image)}

echo "🚀 Starting Training..."
docker run --rm {env_flags} {shlex.quote(image)}
"""

            stdin, stdout, stderr = ssh.exec_command(cmd)
            # Training logs go to stderr; reading them with stdout keeps the SSH buffer from filling up
            stdout.channel.set_combine_stderr(True)

            for line in iter(stdout.readline, ""):
                print(line, end="")

            if stdout.channel.recv_exit_status() != 0:
                print("\n❌ REMOTE SCRIPT FAILED. LOGS FROM USER_DATA:")
                _, log_out, _ = ssh.exec_command("cat /var/log/user-data-debug.log")
                print(log_out.read().decode())

                raise Exception("Training Failed")
                
        finally:
            ssh.close()
    # ------------------------------------------------------------------
    # TASK 6: Cleanup
    # ------------------------------------------------------------------
    terminate_ec2 = EC2TerminateInstanceOperator(
        task_id="terminate_ec2_instance",
        region_name=REGION_NAME,
        instance_ids="{{ task_instance.xcom_pull(task_ids='create_ec2_instance')[0] }}",
        trigger_rule=TriggerRule.ALL_DONE, # Run even if training failed
    )

    # ------------------------------------------------------------------
    # FLOW DEFINITION
    # ------------------------------------------------------------------

    # 1. Instantiate the TaskFlow tasks
    github_signal = wait_for_github_ci()

    # 2. Pass data from Standard Operator -> TaskFlow Task
    # CRITICAL FIX: Use 'create_ec2.output' to get the XComArg (Instance IDs)
    # The operator returns a list, so check_ec2_status will receive that list.
    validated_instance_id = check_ec2_status(create_ec2.output)

    # 3. Pass data between TaskFlow tasks
    public_ip = get_public_ip(validated_instance_id)
    training_output = run_training_via_ssh(public_ip)

    # 4. Define the Execution Order (The Graph)
    # We explicitly chain them to ensure they appear in the UI
    github_signal >> create_ec2

    # Note: The data dependencies above (passing variables) technically imply order,
    # but adding explicit bitshift (>>) guarantees the graph renders correctly.
    create_ec2 >> validated_instance_id >> public_ip >> training_output >> terminate_ec2