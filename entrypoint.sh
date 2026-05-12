#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash

export GZ_VERSION=harmonic

echo "Provided arguments: $@"

exec $@
