#!/bin/bash
# apply_patches.sh
# patches/gr00t/ のファイルを extern/GR00T-WholeBodyControl/ にシンボリックリンクで適用
# extern/ の git 履歴は変更しない

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$REPO_ROOT/patches/gr00t"
EXTERN="$REPO_ROOT/extern/GR00T-WholeBodyControl"

if [ ! -d "$EXTERN" ]; then
    echo "エラー: $EXTERN が見つかりません"
    echo "以下を実行してください:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "=== パッチ適用開始 (シンボリックリンク方式) ==="
echo "PATCH : $PATCH"
echo "EXTERN: $EXTERN"
echo ""

# patches/gr00t/ 以下の全ファイルをシンボリックリンクに差し替え
find "$PATCH" -type f | while read patch_file; do
    # extern/ での対応パスを計算
    rel="${patch_file#$PATCH/}"
    target="$EXTERN/$rel"
    target_dir="$(dirname "$target")"

    # ディレクトリ作成
    mkdir -p "$target_dir"

    # 元ファイルが存在する場合はバックアップ（.orig）
    if [ -f "$target" ] && [ ! -L "$target" ]; then
        cp "$target" "${target}.orig"
        echo "[backup] $rel"
    fi

    # XML ファイルはコピー、その他はシンボリックリンク
    if [[ "$rel" == *.xml ]]; then
        cp "$patch_file" "$target"
        echo "[copy]   $rel"
    else
        ln -sf "$patch_file" "$target"
        echo "[link]   $rel"
    fi
done

echo ""
echo "=== 完了 ==="
echo "元ファイルは .orig として保存されています"
echo "元に戻す場合: bash scripts/remove_patches.sh"
