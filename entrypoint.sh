#!/bin/bash
set -e

if [ "$1" = 'jupyter' ]; then
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
elif [ "$1" = 'clean' ]; then
    python predictor/preprocess/clean_data.py
elif [ "$1" = 'tune' ]; then
    python predictor/model/model.py --tune ${2:--metric}
elif [ "$1" = 'predict' ]; then
    python predictor/model/model.py --predict
elif [ "$1" = 'test' ]; then
    pytest predictor/
else
    python predictor/preprocess/clean_data.py && python predictor/model/model.py --predict
fi
