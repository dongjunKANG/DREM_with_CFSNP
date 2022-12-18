# Variables
SEED=42
PRETRAINED_MODEL_NAME=bert-base-uncased

GPU=3
NUM_EPOCHS=1
LEARNING_RATE=1e-3
LOGGING_STEP=100
WARMUP_RATIO=0.1
BATCH_SIZE=32

FEATURE_TYPE=all
MODE=level
SCORING=yes
LEVEL=5
FREEZE=yes




# Returns to main directory
cd ../

# Train
python3 train_level.py \
    --seed $SEED \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --gpu $GPU \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --logging_step $LOGGING_STEP \
    --warmup_ratio $WARMUP_RATIO \
    --batch_size $BATCH_SIZE \
    --feature_type $FEATURE_TYPE \
    --mode $MODE \
    --level $LEVEL \
    --freeze $FREEZE \
    --scoring $SCORING \