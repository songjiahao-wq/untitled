import time

import onnxruntime as rt
import numpy as  np
import torch

import onnxruntime

# # 创建 ONNX Runtime InferenceSession
# sess = onnxruntime.InferenceSession(r'D:\project\T2M-GPT\models\T2M-GPT-trans.onnx', providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# 准备输入数据（PyTorch张量）
data = torch.randn(1, 512)
print("初始输入",data.shape,'numpy输入:',data.numpy().shape)
print("初始输入",torch.tensor([[1]]).shape,'numpy输入:',torch.tensor([[1]]).numpy().shape)
print('*' * 80)
sess = rt.InferenceSession(r'D:\project\T2M-GPT2\onnx\T2M-GPT-trans2.onnx', providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
iutput_info = sess.get_inputs()
for output in iutput_info:
    print("Iutput Name:", output.name)
    print("Iutput Shape:", output.shape)
output_info = sess.get_outputs()
for output in output_info:
    print("Output Name:", output.name)
    print("Output Shape:", output.shape)

print('*' * 80)
input_names = [input_name.name for input_name in iutput_info ]
output_names = sess.get_outputs()[0].name
# print(len(sess.run([output_names], {input_name:data.astype(np.float32)})))
pred_onx= sess.run([output_names], {'input':data.numpy(), 'idx':torch.tensor([[256,256,417]]).numpy()})[0]#,'idx':torch.tensor([[256, 417, 266, 211, 399]]).numpy()
print("初始输出", pred_onx,pred_onx.shape, '输出numpy类型:', type(pred_onx))
print('*' * 80)


