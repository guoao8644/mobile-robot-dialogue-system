#!/bin/bash
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/config.local.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "エラー: $CONFIG が見つかりません"
    echo "configs/config.yaml をコピーして config.local.yaml を作成してください"
    exit 1
fi

# config.local.yaml から値を読む
export OPENAI_API_KEY=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('openai', {}).get('api_key', ''))
")
export BONES_ROOT=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('bones', {}).get('root', ''))
")

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-api-key-here" ]; then
    echo "エラー: OPENAI_API_KEY が設定されていません"
    echo "$CONFIG に API キーを設定してください"
    exit 1
fi

conda activate g1_deploy 2>/dev/null || true
cd "$REPO_ROOT/src/dialogue_system"

echo "[起動] BONES_ROOT=$BONES_ROOT"
python g1_realtime_dialogue.py "$@"
