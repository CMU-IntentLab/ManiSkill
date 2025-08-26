from dataclasses import dataclass

@dataclass
class LocalArgs:
    wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/aug24_tempckpt/wm_ckpt.pt"
    filter_directory_nogp: str = ''
    filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/aug24_tempckpt/gp_policy.pth'
    
    
