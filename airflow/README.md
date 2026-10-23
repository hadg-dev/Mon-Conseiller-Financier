# Setting up Airflow 2.3.2 with Podman

* Download the docker compose image
```
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.3.2/docker-compose.yaml'
```

* Create the local directories
```
mkdir -p ./dags ./logs ./plugins ./data
chmod -R 777 ./logs ./data
```

* Setting up AIRFLOW_UID
```
echo "AIRFLOW_UID=50000" > .env
```

* Managing Podman permissions
```
systemctl --user enable --now podman.socket
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman/podman.sock"
```

* init the container
```
podman compose up airflow-init
```

* Lauching the app
``` 
podman compose up
```

Then connect to http://localhost:8080  with credentials airflow/airflow