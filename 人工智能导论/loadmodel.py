# 加载整个模型
import torch

loaded_model = torch.load('model/best.pth')
print(loaded_model)
# 或者加载模型的参数
# loaded_params = torch.load('model_params.pth')

# 如果只加载了模型的参数，需要先将参数加载到模型对象中
# 假设我们有一个新的模型实例
# new_model = SimpleModel()
# new_model.load_state_dict(loaded_params)
