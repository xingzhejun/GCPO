import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map
from transformers import AutoProcessor, AutoModel


def initialize_hps_model(args, device):
    preprocess_val = None
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        args.hps_path,
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    
    #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    cp = args.hps_checkpoint_path

    checkpoint = torch.load(cp, map_location=f'cuda:{device}')
    model.load_state_dict(checkpoint['state_dict'])
    processor = get_tokenizer('ViT-H-14')
    reward_model = model.to(device)
    reward_model.eval()
    
    return reward_model, preprocess_val, processor

def initialize_pic_model(args, device):
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    reward_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    return reward_model, processor

def calc_probs(processor, model, prompt, images, device):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        scores = (text_embs @ image_embs.T)[0]
    
    return scores