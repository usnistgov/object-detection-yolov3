#!/bin/bash

screen -dm bash -c "./train.sh '0,1' 1e-4 2>&1 | tee 01.log"
screen -dm bash -c "./train.sh '2,3' 1e-4 2>&1 | tee 23.log"
screen -dm bash -c "./train.sh '4,5' 1e-4 2>&1 | tee 45.log"

