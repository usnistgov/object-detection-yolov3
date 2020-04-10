#!/bin/bash

screen -dm bash -c "./train.sh '0,1' 5e-4"
screen -dm bash -c "./train.sh '2,3' 3e-4"
screen -dm bash -c "./train.sh '4,5' 1e-4"

