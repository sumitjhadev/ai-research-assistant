module.exports = {
  apps: [
    {
      name: 'backend',
      script: 'venv/bin/uvicorn',
      args: 'backend.api:app --host 0.0.0.0 --port 8000',
      cwd: '/home/user/webapp',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1'
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    },
    {
      name: 'frontend',
      script: 'venv/bin/streamlit',
      args: 'run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true',
      cwd: '/home/user/webapp',
      interpreter: 'none',
      env: {
        BACKEND_URL: 'http://localhost:8000'
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
