#!/bin/bash

# Function to run a command and continue even if it fails
run_experiment() {
    echo "Running: $1"
    eval $1
    if [ $? -ne 0 ]; then
        echo "Experiment failed: $1"
    else
        echo "Experiment completed successfully: $1"
    fi
}

# Base training commands
python clear_cache.py
run_experiment "python src/a_unet/train.py -b 16 -e 50 -s 224 -m clip --amp"
python clear_cache.py
run_experiment "python src/a_unet/train.py -b 32 -e 100 --amp"
python clear_cache.py
run_experiment "python src/a_unet/train.py -b 16 -e 100 --amp"
python clear_cache.py
run_experiment "python src/a_unet/train_dropout.py -b 16 -e 100 --amp"
python clear_cache.py
run_experiment "python src/a_unet/train_weigths.py -b 16 -e 100 --amp"
python clear_cache.py

# Run experiments with different optimizers
optimizers=('adam' 'rmsprop' 'sgd')
for opt in "${optimizers[@]}"; do
   run_experiment "python src/a_unet/train.py -b 16 -e 100 --amp --optimizer $opt"
   python clear_cache.py
done

echo "All experiments completed!"
