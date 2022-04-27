# FLASK REDIS API SERVICE

Inspired by https://alimbekov.com/machine-learning-flask-rest-api/  

Version without sklearn, Docker and gunicorn.  

Fixed some bugs.  

Before running flask start the redis server.  

Redis server works only with ubuntu and mac:  
```bash
sudo apt install redis-server  
rq worker rest_api
```