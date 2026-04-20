#!/bin/bash
# remove_patches.sh
# シンボリックリンクを削除して .orig を元に戻す

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$REPO_ROOT/patches/gr00t"
EXTERN="$REPO_ROOT/extern/GR00T-WholeBodyControl"

echo "=== パッチ削除開始 ==="

find "$PATCH" -type f | while read patch_file; do
    rel="${patch_file#$PATCH/}"
    target="$EXTERN/$rel"

    if [ -L "$target" ] || [ -f "$target" ]; then
        if [ -f "${target}.orig" ]; then
            rm "$target"
            mv "${target}.orig" "$target"
            echo "[restored] $rel"
        elif [ -L "$target" ]; then
            rm "$target"
            echo "[removed]  $rel (バックアップなし)"
        fi
    fi
done

echo "=== 完了 ==="
