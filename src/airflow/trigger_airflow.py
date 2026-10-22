import requests
import os
import sys
import json
import logging
from datetime import datetime, timezone

# voir config/airflow.cfg
# deplacer le script dans dags/ ?

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
AIRFLOW_URL = os.getenv("AIRFLOW_URL") # e.g., https://xyz.ngrok-free.app
USERNAME = os.getenv("AIRFLOW_USER")
PASSWORD = os.getenv("AIRFLOW_PASS")
DAG_ID = os.getenv("DAG_ID", "github_ec2_ml_training")
GIT_HASH = sys.argv[1] if len(sys.argv) > 1 else "main"

if not AIRFLOW_URL:
    logging.info("❌ Error: AIRFLOW_URL env var is missing")
    sys.exit(1)

# Common headers (Essential for Ngrok)
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true" # Bypasses Ngrok's landing page
}

def get_jwt_token():
    """
    Step 1: Authenticate and get the JWT Token
    """
    auth_url = f"{AIRFLOW_URL}/auth/token"
    logging.info(f"🔑 Authenticating with {auth_url}...")
    
    try:
        # Airflow 3 usually expects Basic Auth or JSON to issue the token
        # Trying JSON payload first (Standard for JWT endpoints)
        payload = {"username": USERNAME, "password": PASSWORD}
        
        response = requests.post(auth_url, json=payload, headers=HEADERS, timeout=10)
        
        # If JSON body fails, fallback to Basic Auth (legacy compat)
        if response.status_code in [401, 422]:
            logging.info("   -> JSON Auth failed, trying Basic Auth...")
            response = requests.post(auth_url, auth=(USERNAME, PASSWORD), headers=HEADERS, timeout=10)

        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise Exception("No access_token found in response")
        
        logging.info("✅ Authentication Successful")
        return token
        
    except Exception as e:
        logging.info(f"❌ Auth Failed: {e}")
        logging.info(f"Response Body: {response.text if 'response' in locals() else 'None'}")
        sys.exit(1)

def trigger_dag(token):
    """
    Step 2: Trigger the DAG using the Token
    """
    # Note: Airflow 3 API structure might differ slightly (public/v1 vs api/v1)
    # We try the standard V1 endpoint first.
    trigger_url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns"
    
    logging.info(f"🚀 Triggering DAG: {DAG_ID}")
    
    # Add the Bearer Token to headers
    auth_headers = HEADERS.copy()
    auth_headers["Authorization"] = f"Bearer {token}"
    
    # Payload with Git Hash configuration
    current_time = datetime.now(timezone.utc).isoformat()

    payload = {
        "logical_date": current_time,
        "conf": {"git_hash": GIT_HASH},
        "note": "Triggered via GitHub Actions CI"
    }
    try:
        response = requests.post(trigger_url, json=payload, headers=auth_headers, timeout=10)
        response.raise_for_status()
        logging.info(f"✅ Success! Run ID: {response.json().get('dag_run_id')}")
        
    except Exception as e:
        logging.info(f"❌ Trigger Failed: {e}")
        logging.info(f"Response Body: {response.text if 'response' in locals() else 'None'}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jwt = get_jwt_token()
    trigger_dag(jwt)
