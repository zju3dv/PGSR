import os
    
scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path='mip360'
out_base_path='output_mip360'
out_name='test'
gpu_id=0

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f"--quiet -r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--quiet --skip_train"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)
