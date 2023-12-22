import gradio as gr
import torch
from torchvision.transforms import functional as TF
from PIL import Image

from model import LitDyConvSR
from utils import tensor2uint
import config

def RT4KSR_Generate(lr_image):
    print(f'device = {device}')
    litmodel = LitDyConvSR.load_from_checkpoint(
        checkpoint_path="checkpoints/best.ckpt",
        config=config,
        map_location='cuda'
    )
    litmodel.model.to(device)
    litmodel.eval()
    # make width and height divisible by 2
    w, h = lr_image.size
    w -= w % 2
    h -= h % 2
    lr_image = lr_image.resize((w, h))
    
    lr_sample = TF.to_tensor(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_sample = litmodel.predict_step(lr_sample)
        
    sr_sample = tensor2uint(sr_sample * 255.0)
    image_sr_PIL = Image.fromarray(sr_sample)
    
    return image_sr_PIL

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iface = gr.Interface(
        fn=RT4KSR_Generate,
        inputs=[gr.Image(type="pil", label="LR Image", )],
        outputs=[gr.Image(type="pil", label="SR Image")],
        title=f"RT4KSR-Rep-XL Super Resolution Model, device = {device}",
        allow_flagging="never",
        examples=["examples/baby.png", "examples/butterfly.png"]
    )
    iface.launch(share=False)