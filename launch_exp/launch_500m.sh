SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

echo "$SCRIPT_DIR"
echo "$MAIN_DIR"


CONFIG_PATH="$MAIN_DIR/config/500m.yaml"
CMD="python $MAIN_DIR/pretrain/pretrain.py --config_file $CONFIG_PATH"

echo "$CMD"
eval "$CMD"