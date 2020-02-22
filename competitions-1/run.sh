export TRAINING_DATA=input/train_kfolds.csv
export FOLD=0
export MODEL=$1 # VARIABLE TO CHOOSE WHICH MODEL WE WANT TO RUN

python -m src.train