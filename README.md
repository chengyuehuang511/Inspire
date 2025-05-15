# InspireVLA

Official implementation of the paper "[Think Before You Act: Vision-Language-Action Models with Intrinsic Spatial Reasoning]()".

> **Note**: We are doing our best to improve this work. If you have any questions or suggestions, please feel free to create an issue in this repo or contact us at shihan.wu.koorye@outlook.com.

## News

- [2025/5/15] The code is released.

## Introduction

> **Abstract** Leveraging pretrained Vision-Language Models (VLMs) to map language instruction and visual observations to raw low-level actions, Vision-Language-Action models (VLAs) hold great promise for achieving general-purpose robotic systems. Despite their advancements, existing VLAs tend to spuriously correlate task-irrelevant visual features with actions, limiting their generalization capacity beyond the training data. To address this challenge, we propose \textbf{Intrinsic Spatial Reasoning (InSpire)}, which mitigates the adverse effects of spurious correlations by boosting the spatial reasoning ability of VLAs. Specifically, InSpire redirects the model's attention to task-relevant visual clues by simply appending the question *“In which direction is the [object] relative to the robot”* before the language instruction and aligning the VLA's answer *“right/left/up/down/front/back/grasp”* and predicted actions with the ground-truth. Notably, InSpire can be employed as a \textit{plugin} to enhance existing autoregressive VLAs, requiring no extra data or interaction with other large models. Extensive experimental results in both simulation and real-world environments demonstrate the effectiveness and flexibility of our approach.

![Overview](examples/overview.png)
![Method](examples/method.png)

## Experiments

### Overall Performance

**Simulatied Environments**

![Simulated Environments](examples/libero_results.png)

**Real-world Environments**

![Real-world Environments](examples/real_results.png)

## Videos

### Simulated Environments

**MiniVLA**

| Libero-90<br>Butter Drawer | Libero-90<br>Moka Stove | Libero-90<br>Sauce Tray | Libero-90<br>Book Caddy |
|----------------------------|-------------------------|-------------------------|-------------------------|
| ![Libero-90 Butter Drawer](examples/videos/main/libero/baseline/90_butter_drawer.gif) | ![Libero-90 Moka Stove](examples/videos/main/libero/baseline/90_moka_stove.gif) | ![Libero-90 Sauce Tray](examples/videos/main/libero/baseline/90_sauce_tray.gif) | ![Libero-90 Book Caddy](examples/videos/main/libero/baseline/90_book_caddy.gif) |

| Libero-Goal<br>Bowl Plate | Libero-Object<br>Cheese Basket | Libero-Spatial<br>Bowl Plate | Libero-10<br>Book Caddy |
|----------------------------|------------------------------|---------------------------|-------------------------|
| ![Libero-Goal Bowl Plate](examples/videos/main/libero/baseline/goal_bowl_plate.gif) | ![Libero-Object Cheese Basket](examples/videos/main/libero/baseline/object_cheese_basket.gif) | ![Libero-Spatial Bowl Plate](examples/videos/main/libero/baseline/spatial_bowl_plate.gif) | ![Libero-10 Book Caddy](examples/videos/main/libero/baseline/10_book_caddy.gif) |

**InspireVLA (Ours)**

| Libero-90<br>Butter Drawer | Libero-90<br>Moka Stove | Libero-90<br>Sauce Tray | Libero-90<br>Book Caddy |
|----------------------------|-------------------------|-------------------------|-------------------------|
| ![Libero-90 Butter Drawer](examples/videos/main/libero/inspire/90_butter_drawer.gif) | ![Libero-90 Moka Stove](examples/videos/main/libero/inspire/90_moka_stove.gif) | ![Libero-90 Sauce Tray](examples/videos/main/libero/inspire/90_sauce_tray.gif) | ![Libero-90 Book Caddy](examples/videos/main/libero/inspire/90_book_caddy.gif) |

| Libero-Goal<br>Bowl Plate | Libero-Object<br>Cheese Basket | Libero-Spatial<br>Bowl Plate | Libero-10<br>Book Caddy |
|----------------------------|------------------------------|---------------------------|-------------------------|
| ![Libero-Goal Bowl Plate](examples/videos/main/libero/inspire/goal_bowl_plate.gif) | ![Libero-Object Cheese Basket](examples/videos/main/libero/inspire/object_cheese_basket.gif) | ![Libero-Spatial Bowl Plate](examples/videos/main/libero/inspire/spatial_bowl_plate.gif) | ![Libero-10 Book Caddy](examples/videos/main/libero/inspire/10_book_caddy.gif) |

### Real-world Environments

**FAST**

**InspireVLA (Ours)**

## Models Checkpoints

| Model | Dataset | Checkpoint |
|-------|---------|------------|
| MiniVLA | Libero90 | [Download]() |
| InspireVLA | Libero90 | [Download]() |

## Running

1. Clone the repository.

```bash
git clone http://github/user/InspireVLA.git
```

2. Install the dependencies.

```bash
conda create -n inspire python=3.10
conda activate inspire

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-min.txt

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

3. Prepare the dataset.

4. Run the training script.

```bash
bash vla_scripts/train/train_baseline_libero90.sh
bash vla_scripts/train/train_inspire_libero90.sh
```

5. Run the evaluation script.

```bash
bash vla_scripts/eval/eval_baseline_libero90.sh
bash vla_scripts/eval/eval_inspire_libero90.sh
```

## Acknowledgements

Our work is built upon the following open-source projects: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), [Pi-0](https://github.com/Physical-Intelligence/openpi). We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.