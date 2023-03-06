import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    print(torch.version)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()

    x = torch.rand(1, 3, 224, 224)
    mod = torch.jit.trace(model, x)
    mod.save("resnet18_batch1.pt")
    # 构造模型对张量值全为1的输出
    # input_2 = torch.ones((1, 3, 224, 224))
    # output_2 = model(input_2)
    #
    # print('输出张量大小:')
    # print(output_2.shape)
    # print('输出张量保存到csv文件中')
    # np.savetxt('/home/fss/code/kuiper_course/tmp/out.csv', output_2.detach().numpy()[0],
    #            delimiter=',', fmt='%f')

    img = Image.open(r'imgs/c.jpeg')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img)
    # 扩充维度
    input_batch = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    print('-' * 32)
    print('resnet18的输出结果')
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
