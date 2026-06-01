#!/bin/bash
 
docker exec -it ardupilot_sitl /bin/bash -c "mavproxy.py --console --map --aircraft test --master=:14550"
