# VerticalFL

    1. DockerFile
    `docker build -t vertical_fl .`

    2. run 
    docker run -it --name vertical_fl -v C:/Project/VerticalFL/script:/script -v C:/Project/VerticalFL/results:/results -v C:/Project/VerticalFL/data:/data vertical_fl
    docker exec vertical_fl /bin/sh -c "python main.py"


# 
https://github.com/FederatedRL/vertical-FL/blob/master/vertical%20federated%20learning(new).py

# 
https://github.com/zhxchd/vFedCCE