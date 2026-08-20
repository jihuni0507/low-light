# low-light

## Implementation Logic
``` Text
1. Reconstruct Gaussian splats with StereoGS
2. Generate text/image condition vector by CLIP encoder
3. Input Gaussians and condition vector into injection network
4. Freeze Gaussians' geometry, update only SH features
5. Loss - computed based on target SH or current SH

```
## Overall Pipeline
<img width="995" height="963" alt="image" src="https://github.com/user-attachments/assets/f24ab313-1b85-4c14-bfd5-0f7054fb2636" />


## 사용방식
### 🔶 Environment Setting

``` Bash
Set-Location -Path "your/project/directory"
$env:PYTHONPATH = "your/project/directory"
```
### 🔶 `main.py`
`config.yaml` 파일을 통해 실험 세부 설정을 할 수 있다.

**🚫!!`yaml`파일 구성에 따라 달라질 수 있음!!🚫**
- 실험 시나리오 별 `yaml`파일 생성하거나
- Base `yaml` + 세부 `yaml` 모듈화 시도 중

**1) 전체 파이프라인 실행**
``` Bash
export PYTHONPATH=$PWD
python main.py --config config.yaml
```

**2) CLIP만 단독 실행**
``` Bash
export PYTHONPATH=$PWD
python main.py --config config.yaml --stage clip
```

**3) StereoGS base reconstruction**
``` Bash
cd ../StereoGS
python train.py \
  -s /path/to/scene \
  -m ./output/llff/fern_3views \
  --dataset_name LLFF \
  --n_views 3 \
  --resolution 8 \
  --eval \
  --use_mvs_init \
  --mvs_model_name MVSAnywhere \
  --sh_degree 3
```

StereoGS saves the handoff file at
`<model_path>/point_cloud/iteration_<N>/point_cloud.ply`. Set
`injection.gaussian_ply` in `config.yaml` to that file before running the
injection stage. The `--sh_degree 3` option is required because this project
expects 45 non-DC SH values per Gaussian.

**4) SH injection만 실행**
``` Bash
export PYTHONPATH=$PWD
python main.py --config config.yaml --stage injection
```

기본 실행은 injection network를 학습하고 `injection.checkpoint`에 가중치를 저장한다. 저장된 가중치를 사용해 inference만 실행하려면:
``` Bash
python main.py --config config.yaml --stage injection --train False
```
Inference는 체크포인트를 불러와 SH feature를 갱신하고 `injection.output_ply`에 저장한다.

---
**5) Text-Conditioned Training**
``` Bash
python main.py --config config.yaml clip.text_prompt="low-light scene"
```
또는 `config.yaml`을 직접 수정할 수도 있다.

**6) Image-Conditioned Training**
``` Bash
python - <<'PY'
import yaml
from pathlib import Path

p = Path("config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["clip"]["image_path"] = "assets/example.png"
cfg["pipeline"]["run_clip_encode"] = True
cfg["pipeline"]["run_injection"] = True
p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
PY

python main.py --config config.yaml
```

---

### ~~🔶 `train_base_gaussians.py`~~

**One-line Command**
``` Bash
python train/train_base_gaussians.py
```

**Default**
``` Bash
python train/train_base_gaussians.py \
  --colmap_dir datasets/colmap \
  --output_dir ./output/base \
  --epochs 1 \
  --max_train_steps 1000 \
  --learning_rate 1e-3 \
  --device cuda
```
---
### 🔶 Gaussian Reconstruction using StereoGS
Recommended commmand
```bash
python train.py \
  -s /path/to/scene \
  -m ./output/LLFF/scene_3views \
  --dataset_name LLFF \
  --n_views 3 \
  --resolution 8 \
  --eval \
  --sh_degree 3 \
  --use_mvs_init \
  --mvs_model_name MVSAnywhere \
  --opacity_decay \
  --opacity_decay_factor 0.99 \
  --grad_sensitivity 0.5 \
  --generate_binocular_view \
  --start_generate_binocular_view_iters 20000 \
  --cam_trans_dist 4.0 \
  --binocular_model_type FoundationStereo \
  --binocular_depth \
  --lambda_binocular 1.0 \
  --dropout_factor 0.3 \
  --init_pcd_downsample 0.1
```
Required structure of `/path/to/acene`
``` bash
scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```
Example for `office1`:
``` bash
python train.py \
  -s /home/work/test2/datasets/office1 \
  -m ./output/LLFF/office1_3views \
  --dataset_name LLFF \
  --n_views 3 \
  --resolution 8 \
  --eval \
  --sh_degree 3
```
---
### 🔶 `train_injection_network.py`
#### 🔸 Text-Conditioned Training

``` Bash
python train/train_injection_network.py --gaussian_ply path/to/gaussians.ply --text "low-light scene" --condition_mode text --epochs 10
```

#### 🔸 Image-Conditioned Training

``` Bash
python train/train_injection_network.py --gaussian_ply path/to/gaussians.ply --image path/to/input.png --condition_mode image --epochs 10
```
