from tqdm import tqdm
from inference_VQ_Diffusion import VQ_Diffusion


with open('cub_test_captions.txt', 'r') as f:
    captions = f.readlines()
    captions = [caption.strip() for caption in captions]

VQ_Diffusion = VQ_Diffusion(
    config='OUTPUT/pretrained_model/config_text.yaml',
    path='OUTPUT/pretrained_model/cub_pretrained.pth'
)

for caption in tqdm(captions):
    VQ_Diffusion.inference_generate_sample_with_condition(
        caption,
        truncation_rate=0.86,
        save_root="RESULT_cub200_pretrained_nofast",
        batch_size=8,
    )
