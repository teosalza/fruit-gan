import torch
from CGanGen import Generator as cganG
import os 
import torchvision.utils as vutils


model_weight_path = "fuit-gan\\best_saves\\4_fruit_brg_200ep\\netG_final.pth"
device = torch.device("cpu:0")
out_dir = "output"

generator_model = cganG(4,3,45, 100).to(device)
generator_model.load_state_dict(torch.load(model_weight_path))
generator_model.eval()

viz_noise = torch.randn(1,100, device=device)
viz_label = torch.randint(3,4,(1,), device=device)

generated_image = generator_model(viz_noise,viz_label)
vutils.save_image(generated_image, os.path.join(out_dir,'fake_samples_{}.png'.format("inference")))