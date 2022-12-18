# Variables
FEATURE_TYPE=act
MODE=level
LEVEL=3
FREEZE=yes

# Returns to main directory
cd ../

# Train
python3 evaluation.py \
    --feature_type $FEATURE_TYPE \
    --mode $MODE \
    --level $LEVEL \
    --freeze $FREEZE \