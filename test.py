from clip import clip
import torch
from argparse import ArgumentParser

device = 'cuda:6'

a = 'a'
b = 'photo'
c = 'a photo of sth.'

token_1 = clip.tokenize(c).to(device)
print(token_1)

parser = ArgumentParser()
parser.add_argument("--clip-model-name", default='ViT-B/16', type=str)


args = parser.parse_args()

design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2}

args.design_details = design_details

clip_model, clip_image_encoder, fixed_image_encoder, clip_preprocess = clip.load(args, args.clip_model_name, args.design_details, device=device, jit=False)

embedding_1 = clip_model.token_embedding(token_1)

print(embedding_1)
print(embedding_1[0][:5])

x = torch.tensor([[1,2,3,4], [1,2,3,5], [1,6,2,3], [1,3,2,8], [9, 1, 2, 3]])
print(x[torch.arange(x.shape[0]), x.argmax(dim=-1)])

# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # 假设有一个常数张量
#         const_tensor = torch.tensor([1.0, 2.0, 3.0])
#         # 使用 register_buffer 注册为模型的缓冲区
#         self.register_buffer("my_constant", const_tensor)

# # 创建模型
# model = MyModel()

# # 模型的缓冲区会随着模型一起保存和加载
# torch.save(model.state_dict(), "model.pth")
# loaded_model = MyModel()
# loaded_model.load_state_dict(torch.load("model.pth"))

# # my_constant 不是一个参数，但在保存和加载时会被正确处理
# print(loaded_model.my_constant)

# a = torch.tensor([[1,2,3,4], [1,2,3,4]])
# b = torch.tensor([[5,6,7,8,9], [5,6,7,8,9]])
# print(a.shape)
# print(b.shape)
# c = torch.cat((a, b), dim=-1)
# print(c)
# print(c.shape)