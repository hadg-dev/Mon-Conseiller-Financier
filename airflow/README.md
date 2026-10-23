# Setting up Airflow 3.3.2 with Podman

* Download the docker compose image
```commandline
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.3.2/docker-compose.yaml'
```

* Create the local directories
```commandline
mkdir -p ./dags ./logs ./plugins ./config
sudo chmod -R 777 ./logs ./data ./config
```

* Setting up AIRFLOW_UID
```commandline
echo "AIRFLOW_UID=50000" > .env
```

* Managing Podman permissions
```commandline
systemctl --user enable --now podman.socket
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
```

* init airflow.cfg with default values
```commandline
podman compose run airflow-cli airflow config list
```

* init container database
```commandline
podman compose up airflow-init
```

* Lauching the app
```commandline
podman compose up
```

Then connect to http://localhost:8080  with credentials airflow/airflow