# T²Mem project website

Interactive presentation of **T²Mem: Learning Test-Time Memory for Robotics**.

The site explains observation-grounded memory, fast-weight read/write updates,
alternating training, and selected results from the paper.

## Preview

```sh
python3 -m http.server 8765
```

Open http://localhost:8765. No build step or package installation is required.

## Contents

- `index.html`: research narrative and page structure
- `style.css`: responsive layout and data-flow animations
- `app.js`: step controls, training stages, result filters and memory interventions
- `assets/`: paper figures and website icon

The fast-weight animation is schematic. Numerical results come from the supplied
manuscript's main table, experiments section, ablation figure, and latency figure.
The main table and task-specific interventions use distinct evaluation protocols,
identified beside the corresponding visualizations. Baseline scores are quoted
by the manuscript from [RoboMME](https://arxiv.org/abs/2603.04639).

GitHub Pages serves this repository's `main` branch. Text uses Google Sans and Noto Sans
from Google Fonts, with system-font fallbacks.

The three task demonstration videos are embedded from the RoboMME project site
(VideoUnmask, PickXtimes, and PatternLock, whose source asset is DrawPattern.mp4).
They illustrate the benchmark tasks and are not T²Mem policy evaluation rollouts.
Playback requires network access to robomme.github.io.
