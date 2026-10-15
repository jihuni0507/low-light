### 🟦 Checklist

- [ ] **Dark Image 생성 로직 점검** - `enhance_dark.py`, `generate_dark.py` 로직 및 성능 비교
- [ ] JPG 이미지 darken 이후 zero-dce로 복원했을 때 결과가 이상적이지 않음, PNG 이미지로 했을 때도 그런지 확인

- [ ] **Per-scene Optimization** 가장 먼저 10장 이상의 enhanced image 바탕 3DGS로 견고하게 geometry를 재건한 이후에(geometry는 잘 되었다는 가정) 원래대로 두장의 이미지만 입력으로 넣었다고 생각하고 가우시안 컬러 수정하는 네트워크 검증