## 🔶 Training Injection Network

``` bash
cd /home/work/test2/low-light
export PYTHONPATH=$PWD

python train/train_injection_network.py \
  --dataset_yaml dataset.yaml \
  --sh_degree 3 \
  --device cuda \
  --epochs 10 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --hidden_dim 256 \
  --gaussian_chunk_size 16384 \
  --lambda_sh 0 \
  --lambda_image 1.0 \
  --lambda_ssim 0.2 \
  --lambda_well_exposure 0.1 \
  --lambda_well_saturated 0.1 \
  --lambda_gain 0.001 \
  --lambda_spatial_smooth 0.01 \
  --lambda_magnitude 0.001 \
  --spatial_smooth_k 16 \
  --save_dir ./output/injection
```

 - 초기에는 `lambda_well_exposure, lambda_well_saturated, lambda_gain` 값을 작게 주고 이후에 loss를 강하게 줄 수 있다
 - `lambda_sh = 0`은 현새 pseudo-GT 구조에서 필수