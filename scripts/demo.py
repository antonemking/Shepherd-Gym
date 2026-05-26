"""Quick demo: benchmark the scripted baselines and emit comparison clips.

    python scripts/demo.py            # default 30 eval seeds
    python scripts/demo.py 60         # more seeds
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shepherd_gym.benchmark import main

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(out_dir=str(pathlib.Path(__file__).resolve().parents[1] / "out"), episodes=episodes)
