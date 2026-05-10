#!/bin/bash

set -e

source /opt/ros/humble/setup.bash

export GZ_VERSION=harmonic

echo "Provided arguments: $@"

exec $@
