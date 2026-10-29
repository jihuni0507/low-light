import numpy as np
import torch
from plyfile import PlyData

def main():
    ply_path = r"C:\Users\IVL-SERVER2\Desktop\jihun\low-light\data\cabin_undistorted\3dgs_gaussians.ply"
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    num_points = len(vertex)
    
    property_names = [p.name for p in vertex.properties]
    print(f"=== [PLY 파일 정보] ===")
    print(property_names)
    print(f"총 가우시안 개수 (N): {num_points:,}")
    print(f"전체 속성 개수: {len(property_names)}")
    
    # 3. 속성별 numpy 배열 추출 및 PyTorch 텐서 변환

    # (1) 위치 (x, y, z) -> Shape: (N, 3)
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    xyz_tensor = torch.from_numpy(xyz).float()

    # (2) 불투명도 (opacity) -> Shape: (N, 1)
    opacity = np.asarray(vertex["opacity"])[..., np.newaxis]
    opacity_tensor = torch.from_numpy(opacity).float()

    # (3) 스케일 (scale_0, scale_1, scale_2) -> Shape: (N, 3)
    scale_names = [f"scale_{i}" for i in range(3)]
    scale = np.stack([vertex[p] for p in scale_names], axis=-1)
    scale_tensor = torch.from_numpy(scale).float()

    # (4) 회전 쿼터니언 (rot_0, rot_1, rot_2, rot_3) -> Shape: (N, 4)
    rot_names = [f"rot_{i}" for i in range(4)]
    rot = np.stack([vertex[p] for p in rot_names], axis=-1)
    rot_tensor = torch.from_numpy(rot).float()

    # (5) 0차 SH 계수 (RGB Base Color) -> Shape: (N, 3)
    features_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
    )
    features_dc_tensor = torch.from_numpy(features_dc).float()

    # (6) 고차 SH 계수 (f_rest_0 ~ f_rest_44) -> Shape: (N, 45) -> Reshape to (N, 15, 3)
    extra_sh_names = [
        p for p in property_names if p.startswith("f_rest_")
    ]
    if extra_sh_names:
        # f_rest_0 ~ f_rest_44 순서대로 정렬하여 추출
        extra_sh_names = sorted(
            extra_sh_names, key=lambda x: int(x.split("_")[-1])
        )
        features_extra = np.stack([vertex[p] for p in extra_sh_names], axis=-1)
        features_extra_tensor = torch.from_numpy(features_extra).float()

        # (N, 45) -> (N, 15, 3) 또는 (N, 3, 15)로 변환 가능
        features_extra_reshaped = features_extra_tensor.reshape(
            num_points, 15, 3
        )
    else:
        features_extra_tensor = None
        features_extra_reshaped = None

    # (7) 모든 속성을 하나로 연결(Concatenate) -> Shape: (N, 62)
    # 위치(3) + DC SH(3) + Extra SH(45) + Opacity(1) + Scale(3) + Rotation(4)
    concatenated_list = [xyz, features_dc]
    if extra_sh_names:
        concatenated_list.append(features_extra)
    concatenated_list.extend([opacity, scale, rot])

    all_features = np.concatenate(concatenated_list, axis=-1)
    all_tensor = torch.from_numpy(all_features).float()

    # 4. Shape 출력 확인
    print("\n=== [텐서 Shape 확인] ===")
    print(f"Positions (xyz)        : {xyz_tensor.shape}")
    print(f"Opacity (opacity)      : {opacity_tensor.shape}")
    print(f"Scales (scale)         : {scale_tensor.shape}")
    print(f"Rotations (quaternion) : {rot_tensor.shape}")
    print(f"SH DC (base color)     : {features_dc_tensor.shape}")
    if features_extra_tensor is not None:
        print(f"SH Rest (flat 45)      : {features_extra_tensor.shape}")
        print(f"SH Rest (reshaped)     : {features_extra_reshaped.shape}")
    print(f"Concatenated Full Tensor: {all_tensor.shape}")

    return {
        "xyz": xyz_tensor,
        "opacity": opacity_tensor,
        "scale": scale_tensor,
        "rotation": rot_tensor,
        "sh_dc": features_dc_tensor,
        "sh_rest": features_extra_reshaped,
        "full": all_tensor,
    }
    
    # 5. Additional
    print("\n=== [추가 정보] ===")
    centroid = xyz.mean(0)
    for camera in cameras:
        c2w = np.array(camera["c2w"])
        p_cam = np.linalg.inv(c2w) @ [centroid[0], centroid[1], centroid[2], 1.0]
        print(p_cam)
    

if __name__ == "__main__":
    main()