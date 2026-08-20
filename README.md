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

### 🔶 Dataset 구성 가이드

Injection network 학습용 데이터셋은 하나의 입력 Gaussian과 여러 개의
restyled target Gaussian을 연결하는 구조로 구성한다.

```text
source Gaussian PLY + input image + text prompt
                         |
                         v
                 target Gaussian PLY
```

각 샘플은 다음 세 가지 정보를 가진다.

| 항목 | 설명 |
| --- | --- |
| `input_image` | 입력 장면의 원본 이미지 경로 |
| `target_gs` | 입력 이미지에 대해 Synthesized re-styled image를 Gaussian splatting한 target PLY 경로 |
| `prompt` | target 이미지를 생성할 때 사용한 텍스트 prompt |

입력 Gaussian PLY는 동일한 장면을 기준으로 여러 prompt를 학습할 수 있도록
`dataset.source_gaussian_ply`에 공통 경로로 기록한다. Target PLY는 각 샘플의
`target_gs`에 기록한다.

권장 디렉터리 구조:

```text
low-light/
├── dataset.yaml
└── datasets/
    ├── inputs/
    │   └── office1.png
    └── targets/
        ├── office1_daylight/
        │   └── point_cloud.ply
        └── office1_low_light/
            └── point_cloud.ply
```

`dataset.yaml` 예시:

```yaml
dataset:
  name: low_light_injection
  root: "."
  source_gaussian_ply: "../StereoGS/output/LLFF/office1_3views/point_cloud/iteration_30000/point_cloud.ply"
  samples:
    - id: office1_daylight
      input_image: "datasets/inputs/office1.png"
      target_gs: "datasets/targets/office1_daylight/point_cloud.ply"
      prompt: "a brightly lit daytime scene"
    - id: office1_low_light
      input_image: "datasets/inputs/office1.png"
      target_gs: "datasets/targets/office1_low_light/point_cloud.ply"
      prompt: "a low-light indoor scene"
```

데이터를 등록할 때 다음 조건을 확인한다.

- 모든 경로는 `dataset.root`를 기준으로 해석한다.
- 입력 Gaussian과 target Gaussian의 개수가 동일해야 한다.
- 두 PLY의 SH degree와 SH feature shape이 동일해야 한다.
- Target PLY는 실제 synthesized image를 Gaussian splatting한 결과여야 한다.
- 입력 이미지와 prompt는 target 이미지를 생성한 조건과 정확히 대응해야 한다.
- `output/injection/injected_gaussians.ply` 같은 inference 결과물을 target으로 재사용하지 않는다.

현재 `dataset.yaml`은 데이터 포맷을 정의하는 단계이며, dataset 전체를 읽는
DataLoader는 `train/injection_dataset.py`에 구현되어 있다. 실제 학습에서는
source Gaussian의 SH feature와 각 샘플의 target SH feature 사이의 loss를
계산한다.

Dataset과 DataLoader 사용 예시:

```python
from train.injection_dataset import build_injection_dataloader, encode_prompt_batch

dataloader = build_injection_dataloader(
  "dataset.yaml",
  sh_degree=3,
  batch_size=2,
  shuffle=True,
  num_workers=0,
  device="cpu",
)

for batch in dataloader:
  condition = encode_prompt_batch(batch, clip_encoder, device="cuda")
  for sample, sample_condition in zip(batch, condition):
    source_features = sample["source_features"].to("cuda")
    target_features = sample["target_features"].to("cuda")
    predicted = injection_network(source_features, sample_condition)
```

Gaussian 개수가 sample마다 다를 수 있으므로 기본 `collate_fn`은 tensor를
하나로 쌓지 않고 sample dictionary의 list를 반환한다. 따라서 각 sample의
loss를 계산한 뒤 평균을 내는 방식으로 학습한다. 모든 sample의 Gaussian 수가
같은 경우에도 이 방식은 사용할 수 있으며, 추후 padding 기반 batch 연산으로
최적화할 수 있다.

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
