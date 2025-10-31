import torch
#from vit_pytorch.vit_for_small_dataset import ViT
from vit_pytorch import MAE, ViT
import torch.optim as optim
from torch.optim import lr_scheduler

v = ViT(
    image_size = 512,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    #dropout = 0.1,
    #emb_dropout = 0.1
)

mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

opt = optim.AdamW(mae.parameters(), lr=0.00015, betas=(0.9, 0.95),weight_decay=0.05)

cosine_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, 30)


for i in range(50):
    print(i)
    print(cosine_lr_scheduler.get_last_lr())

    for j in range(2):
        images = torch.randn(8, 3, 512, 512)
        #loss = mae(images)
        #loss.backward()
        #opt.step()

    cosine_lr_scheduler.step()
# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

# save your improved vision transformer
torch.save(v.state_dict(), './trained-vit.pt')