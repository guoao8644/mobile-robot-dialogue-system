# mobile-robot-dialogue-system

Unitree G1 ロボット向けリアルタイム音声対話システム。  
OpenAI Realtime API + BONES-SEED モーション + GR00T WBC を統合。

---

## 構成

```
extern/
  GR00T-WholeBodyControl/     # NVlabs submodule (deploy + sim)
  kimodo/                     # nv-tlabs submodule (motion generation)
patches/gr00t/                # GR00T への変更パッチ（シンボリックリンクで適用）
src/dialogue_system/          # 音声対話システム本体
scripts/                      # 起動ラッパースクリプト
configs/                      # 設定ファイル
environments/                 # 環境セットアップメモ
data/                         # モーションデータ等
```

---

## セットアップ

### Step 1: リポジトリのクローン

```bash
git clone <this-repo>
cd mobile-robot-dialogue-system
git submodule update --init --recursive
cd extern/GR00T-WholeBodyControl
git lfs pull
```

---

### Step 2: Deploy 環境のセットアップ（GR00T WBC）

公式ドキュメント: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html



#### A: TensorRT のインストール

> ⚠️ **重要:** 必ず指定バージョンの TensorRT を使用してください。異なるバージョンを使用するとプランナーが誤った動作を出力し、**ロボットの危険な動作を引き起こす可能性があります。**

NVIDIA Developer から TAR パッケージ（DEB ではなく）をダウンロードしてください。
アーカイブは約 10GB です。

```bash
sudo apt-get install -y pv
pv TensorRT-*.tar.gz | tar -xz -f -
mv TensorRT-* ~/TensorRT
echo 'export TensorRT_ROOT=$HOME/TensorRT' >> ~/.bashrc
source ~/.bashrc
```

---

#### B: Native環境の構築

```bash
cd extern/GR00T-WholeBodyControl/gear_sonic_deploy

# システム依存関係のインストール
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

# 環境セットアップ
source scripts/setup_env.sh

# 永続化する場合
echo "source $(pwd)/scripts/setup_env.sh" >> ~/.bashrc

# ビルド
just build
```

#### C: Docker (ROS2 開発環境)
```bash
# Downloading Model Checkpoints
pip install huggingface_hub
cd extern/GR00T-WholeBodyControl
python download_from_hf.py
```

```bash
# Docker グループへの追加（初回のみ）
sudo usermod -aG docker $USER
newgrp docker

# TensorRT パスを設定（~/.bashrc に追加）
export TensorRT_ROOT=/path/to/TensorRT

# コンテナ起動
cd extern/GR00T-WholeBodyControl/gear_sonic_deploy
./docker/run-ros2-dev.sh

# コンテナ内で
source scripts/setup_env.sh
just build
```

#### D: Simulator 環境のセットアップ（MuJoCo Sim）

```bash
cd extern/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
```

これで `.venv_sim` が作成されます。


### Step 3: パッチの適用

GR00T への変更をシンボリックリンクで適用します。

```bash
cd mobile-robot-dialogue-system
bash scripts/apply_patches.sh
```

> **注意:** `git submodule update` 実行後は再度 `apply_patches.sh` を実行してください。

---

### Step 4: BONES-SEED データセットの配置

BONES-SEED データセットから `g1.tar.gz` をダウンロードし、`data/` 配下に展開してください。  
ダウンロード方法は Hugging Face のデータセットページを参照してください:

https://huggingface.co/datasets/bones-studio/seed

```bash
cd mobile-robot-dialogue-system
mkdir -p data

# 例: ダウンロードした g1.tar.gz を data/ に配置してから展開
tar -xzf /path/to/g1.tar.gz -C data/
```

---

### Step 5: Dialogue 環境のセットアップ

```bash
# conda 環境を使用（g1_deploy）
conda activate g1_deploy

pip install openai websockets pyaudio pyzmq scipy pandas numpy pillow pyyaml
```

---

### Step 6: 設定ファイルの作成

```bash
cp configs/config.yaml configs/config.local.yaml
```

`configs/config.local.yaml` を編集して API キーを設定：

```yaml
openai:
  api_key: "sk-..."

bones:
  root: "/path/to/g1/csv"
```

> **注意:** `config.local.yaml` は `.gitignore` により git に上がりません。

---

## 起動

3つのターミナルで順番に起動します。

### Terminal 1: MuJoCo シミュレーター

```bash
bash scripts/run_sim.sh
```

起動後は MuJoCo ウィンドウ側で以下の操作を行います。

- `9` キー: 吊り上げ状態の G1 を床に下ろします
- `Back` キー: 初期状態に戻します
- `C` キー: 視点を切り替えます
  `robot` 視点、`user` 視点、全体視点、自由視点を順番に切り替えます
- `P` キー: 追加の camera viewer ウィンドウを開閉します
- `]` キー: viewer に表示する映像を切り替えます
  `ego_view` と `user_eye` を切り替えます
- `Space` キー: user model の向きを 180 度切り替えます
- 矢印キー: MuJoCo 内の user model を移動します

`P` キーで開く viewer ウィンドウの下部には、User が Robot を注視しているかどうかの信号が表示されます。  
この映像と `robot_in_user_view` 信号は ZMQ で配信されています。

### Terminal 2: Deploy (WBC)

```bash
bash scripts/run_deploy.sh
```

Deploy 側では、起動後に `Y` キーを押して deploy を開始してください。  
その後、ターミナルに `Init Done` と表示されてから次の `run_dialogue.sh` を起動します。  
`Init Done` が出る前に Dialogue を起動しないでください。

### Terminal 3: 音声対話システム

```bash
bash scripts/run_dialogue.sh
```

起動後は音声対話に加えて、キーボードで手動操作もできます。

- `W`: 前進
- `S`: 後退
- `A`: 左旋回
- `D`: 右旋回
- `Space`: 停止

### 停止方法

- `run_sim.sh` を実行しているターミナル: `Ctrl + C`
- `run_deploy.sh` を実行しているターミナル: `O`
- `run_dialogue.sh` を実行しているターミナル: `Ctrl + C`

---

## キー操作（MuJoCo ウィンドウ）

| キー | 機能 |
|------|------|
| `C` | カメラ切替 (tracking / overview / front / user_eye) |
| `]` | ストリームカメラ切替 (ego_view ↔ user_eye) |
| `P` | camera viewer ウィンドウ 開閉 |
| `Space` | User モデル 180度回転 |
| 矢印キー | User モデル移動 |
| `Home` | User モデルを G1 前方にリセット |

---

## パッチ管理

```bash
# パッチ適用
bash scripts/apply_patches.sh

# パッチ削除（元に戻す）
bash scripts/remove_patches.sh
```

---

## 環境一覧

| 環境 | 用途 | 場所 |
|------|------|------|
| `.venv_sim` | MuJoCo シミュレーター | `extern/GR00T-WholeBodyControl/.venv_sim` |
| `conda: g1_deploy` | Deploy + Dialogue | conda 環境 |

---

## Kimodo による Motion 生成

Kimodo を用いた motion 生成手順は今後追記予定です。

---

## Dialogue System 構成

現在の Dialogue System のメイン実装は [`g1_realtime_dialogue.py`](</home/unitree-g1/Documents/G1/mobile-robot-dialogue-system/src/dialogue_system/g1_realtime_dialogue.py:1>) です。

- 音声対話モデルには OpenAI Realtime API を使用しています
  現在の接続先モデルは `gpt-4o-realtime-preview` です
- 入力はマイク音声、出力は音声応答とテキストです
  マイク音声を Realtime API にストリーミング送信し、返ってきた音声をスピーカーで再生します
- 会話中のロボット動作は Realtime API の function calling を使って制御しています
  `select_motion` で会話内容に合うモーションを選びます
  `walk_command` で前進、後退、旋回、停止を指示します

### 動作の実装

- 会話用モーションは BONES-SEED の CSV を読み込んで使います
- CSV の関節角は MuJoCo 順序から IsaacLab / SONIC 順序へ並べ替えています
- BONES-SEED の 120fps データは SONIC 側に合わせて 50fps に線形補間しています
- 補間後に関節速度 `joint_vel` と姿勢 `body_quat` を作成し、ZMQ で `pose` トピックへ送信します
- この送信処理は [`src/motion_generation/bones_to_sonic.py`](</home/unitree-g1/Documents/G1/mobile-robot-dialogue-system/src/motion_generation/bones_to_sonic.py:1>) と同じロジックをベースにしています

### 歩行の実装

- 歩行は CSV モーション再生ではなく、SONIC の planner コマンドを ZMQ で送って制御しています
- `forward`、`backward`、`turn_left`、`turn_right`、`stop` をサポートしています
- 対話中に「3歩前へ」のような指示があった場合は、`steps` パラメータで複数回の歩行を実行します
- `run_dialogue.sh` 起動後は `W` `S` `A` `D` `Space` でも同じ歩行制御を手動で実行できます

### 対話と動作の関係

- モデルは返答前に `select_motion` を呼ぶようにプロンプトで制約しています
- ユーザーが話し始めたときは、再生中のモーションを中断します
- モーション再生時は planner モードから streaming モードへ切り替え、再生後に planner モードへ戻します
- そのため、会話応答、ジェスチャー、歩行を 1 つの対話ループの中で統合しています

---


## MuJoCo シーン設定

現在の MuJoCo シーン設定は [`scene_43dof.xml`](</home/unitree-g1/Documents/G1/mobile-robot-dialogue-system/extern/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml:1>) で定義されています。

- ベースとなるロボットモデルは `g1_29dof_with_hand.xml` を読み込んでいます
- 床は plane geom で定義されています
- 壁は 8m x 6m の長方形領域になるように配置されています
  中心は `(0, 0)`、x 方向は `±4m`、y 方向は `±3m` です
  壁の厚さは `0.1m`、高さは `1.2m` です
- 障害物として 3 つのボックスを配置しています
  赤い箱: `0.4m x 0.4m x 0.5m`、位置は `(2.0, 1.0, 0.25)`
  青い箱: `0.3m x 0.6m x 0.4m`、位置は `(-1.5, -1.5, 0.2)`
  緑の箱: `0.5m x 0.5m x 0.3m`、位置は `(1.0, -2.0, 0.15)`
- 固定カメラとして `overview` と `front` を配置しています
  `overview` は空間全体を斜め上から見る俯瞰カメラです
  `front` はロボット正面を見る固定カメラです
- `user` mocap body もシーン内に含まれています
  初期位置は `(1.5, 0, 0.9)` です
  user body には `user_eye` カメラが付いており、viewer や ZMQ 配信映像の視点切替に使われます
- offscreen レンダリング用の framebuffer は `1024 x 1024` に設定しています
  現在の `ego_view` / `user_eye` の配信サイズ `512 x 512` に対応するためです

壁や障害物、固定カメラ、user model の初期位置を変更したい場合は、上記の `scene_43dof.xml` を編集してください。
