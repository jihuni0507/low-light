import argparse
import os
import subprocess
import sys


def main():
  # 1. 터미널 인자 설정 (argparse)
  parser = argparse.ArgumentParser(
      description="파이썬으로 COLMAP 자동화 파이프라인 실행하기"
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      required=True,
      help="멀티뷰 이미지 폴더가 있는 부모 디렉토리 경로",
  )
  parser.add_argument(
      "--scene_name", type=str, required=True, help="씬 이름 (예: object)"
  )

  args = parser.parse_args()

  # 데이터베이스 경로 설정
  database_dir = os.path.join(args.data_dir, args.scene_name)

  # 전달받은 인자 확인 (수정됨)
  print(
      f"[*] 작업 디렉토리: {args.data_dir}, 씬 이름: {args.scene_name} 작업"
      " 시작..."
  )
  print(f"[*] Database 경로: {database_dir}.db")

  # 2. 실행할 Bash 커맨드 목록 정의
  # 괄호()로 묶어주면 백슬래시 없이 안전하게 여러 줄로 작성할 수 있습니다.
  commands = [
      # 1. Feature Extraction
      (
          f"QT_QPA_PLATFORM=offscreen colmap feature_extractor"
          f" --database_path {database_dir}.db"
          f" --image_path {args.data_dir}/images"
      ),
      # 2. Feature Matching
      f"colmap exhaustive_matcher --database_path {database_dir}.db",
      # 3. Sparse Reconstruction
      f"mkdir -p {args.data_dir}/sparse",
      (
          f"colmap mapper"
          f" --database_path {database_dir}.db"
          f" --image_path {args.data_dir}/images"
          f" --output_path {args.data_dir}/sparse"
      ),
      # 4. Binary to Text
      (
          f"colmap model_converter"
          f" --input_path {args.data_dir}/sparse/0"
          f" --output_path {args.data_dir}/sparse/0"
          f" --output_type TXT"
      ),
      # 5. Image Undistortion
      f"mkdir -p {args.scene_name}_undistorted",
      (
          f"colmap image_undistorter"
          f" --image_path {args.data_dir}/images"
          f" --input_path {args.data_dir}/sparse/0"
          f" --output_path {args.scene_name}_undistorted"
          f" --output_type COLMAP"
      ),
      (
          f"colmap model_converter"
          f" --input_path {args.scene_name}_undistorted/sparse"
          f" --output_path {args.scene_name}_undistorted/sparse"
          f" --output_type TXT"
      ),
      # 6. Generate PLY point cloud file
      (  
          f"colmap model_converter"
          f" --input_path {args.scene_name}_undistorted/sparse"
          f" --output_path {args.scene_name}_undistorted/sparse/points.ply"
          f" --output_type PLY"
      )
  ]

  # 3. 커맨드를 순차적으로 실행
  for cmd in commands:
    print(f"\n[실행 중] $ {cmd}")
    try:
      result = subprocess.run(cmd, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
      print(f"[오류 발생] 커맨드 실패: {cmd}", file=sys.stderr)
      print(f"에러 코드: {e.returncode}", file=sys.stderr)
      sys.exit(1)

  print("\n[*] 모든 COLMAP 파이프라인 작업이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
  main()