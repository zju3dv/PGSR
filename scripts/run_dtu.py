import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='dtu'
out_base_path='output_dtu'
eval_path='dtu_eval'
out_name='test'
gpu_id=0

for scene in scenes:
    
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/dtu_scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} -r2 --ncc_scale 0.5'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_dtu.py -m {out_base_path}/dtu_scan{scene}/{out_name}'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/dtu_eval.py --data {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion.ply --scan {scene} --mode mesh --dataset_dir {eval_path} --vis_out_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh'
    print(cmd)
    os.system(cmd)