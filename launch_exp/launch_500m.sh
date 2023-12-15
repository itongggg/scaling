SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

echo "$SCRIPT_DIR"
echo "$MAIN_DIR"


CONFIG_PATH="$MAIN_DIR/config/lowrank-kl.yaml"
CMD="python $MAIN_DIR/pretrain/stage_1.py --config_file $CONFIG_PATH"

echo "$CMD"
eval "$CMD"