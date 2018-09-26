#!/bin/sh

libbi filter \
  --model-file KalmanFilter.bi \
  --obs-file data/sp500.nc \
  --filter bootstrap \
  --start-time 0 \
  --end-time 100 \
  --nparticles 43210 \
  --output-file filtered.nc
