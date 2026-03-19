pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        PYTHON_BIN = 'python3'
        VENV_DIR = '.venv'
        IMAGE_TAG = "${env.BRANCH_NAME ?: 'manual'}-${env.BUILD_NUMBER}"
        DEPLOY_PATH = '/opt/tcb-fraud-detection-mlops'
        DEPLOY_REF = "${env.BRANCH_NAME ?: 'main'}"
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    ${PYTHON_BIN} -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install flake8 pytest pytest-cov
                    pip install -r ml_pipeline/requirements.txt -r serving_api/requirements.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    flake8 ml_pipeline/src serving_api/app dags
                '''
            }
        }

        stage('Unit And Integration Tests') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pytest ml_pipeline/tests/test_preprocess.py \
                      --cov=ml_pipeline.src.preprocess \
                      --cov-report=term-missing \
                      --cov-fail-under=80
                    pytest serving_api/tests \
                      --cov=serving_api.app.main \
                      --cov=serving_api.app.model_loader \
                      --cov-report=term-missing \
                      --cov-fail-under=80
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build \
                      --file serving_api/Dockerfile \
                      --tag tcb-fraud-fastapi:${IMAGE_TAG} \
                      .
                    mkdir -p build
                    printf '%s' ${IMAGE_TAG} > build/image-tag.txt
                '''
                archiveArtifacts artifacts: 'build/image-tag.txt', fingerprint: true
            }
        }

        stage('Deploy To Google Cloud VPS') {
            when {
                anyOf {
                    branch 'main'
                    branch 'dev/ver2'
                }
            }
            steps {
                sshagent(credentials: ['gcp-vps-ssh']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} \
                          "mkdir -p ${DEPLOY_PATH} && \
                           cd ${DEPLOY_PATH} && \
                           if [ ! -d .git ]; then git clone ${GIT_URL} .; fi && \
                           git fetch --all --prune && \
                           git checkout ${DEPLOY_REF} && \
                           git pull origin ${DEPLOY_REF} && \
                           if [ ! -f .env ]; then cp .env.example .env; fi && \
                           IMAGE_TAG=${IMAGE_TAG} docker compose up -d --build"
                    '''
                }
            }
        }

        stage('Post-Deploy Health Checks') {
            when {
                anyOf {
                    branch 'main'
                    branch 'dev/ver2'
                }
            }
            steps {
                sshagent(credentials: ['gcp-vps-ssh']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} \
                          "cd ${DEPLOY_PATH} && \
                           set -a && \
                           . ./.env && \
                           set +a && \
                           for url in \
                             http://localhost:\${FASTAPI_PORT}/health \
                             http://localhost:\${MLFLOW_PORT} \
                             http://localhost:\${AIRFLOW_PORT}/health \
                             http://localhost:\${GRAFANA_PORT}/api/health; do \
                             echo \"Checking \${url}\"; \
                             for attempt in 1 2 3 4 5 6 7 8 9 10; do \
                               if curl --fail --silent --show-error \${url} > /dev/null; then \
                                 break; \
                               fi; \
                               sleep 10; \
                             done; \
                             curl --fail --silent --show-error \${url} > /dev/null; \
                           done"
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'rm -rf ${VENV_DIR}'
        }
    }
}
