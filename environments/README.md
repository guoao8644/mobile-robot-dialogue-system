# 環境構成

## 環境1: venv_sim (MuJoCo シミュレーター用)
- Python 3.10
- 場所: `extern/GRooT-WholeBodyControl/.venv_sim`
- 用途: `run_sim_loop.py` の実行

```bash
source extern/GRooT-WholeBodyControl/.venv_sim/bin/activate
pip install -r environments/requirements_sim.txt
```

## 環境2: conda g1_deploy (deploy + dialogue 用)
- Python 3.13
- 場所: conda 環境 `g1_deploy`
- 用途: `deploy.sh`, `g1_realtime_dialogue.py`, `g1_walk.py` の実行

```bash
conda activate g1_deploy
pip install -r environments/requirements_deploy.txt
```
