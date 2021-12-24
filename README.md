# Pytorch-Onnx-DLC
 A code generator from PyTorch to Onnx to Qualcomm .dlc for Snapdragon chips. Convert Mobilenetv1_SSD from Pytorch to ONNX then to .dlc . Pytorch code borrowed from https://github.com/qfgaohao/pytorch-ssd
## PYTORCH TO ONNX 
- First convert Pytorch .pth to onnx. (preferably remove Relu). Watch example at convert_to_caffe2_models.py
- Simplify onnx and convert using:
>python3 -m onnxsim mb1-ssd.onnx mb1-ssd-sim.onnx
>
- Check output using netron and load onnx to test output. See run_onnx.py. Remember onnx use input size of 1,3,300,300.
## ONNX TO DLC 
- Remember .dlc use input of size 1,300,300,3
- Covert from onnx to dlc using 
 >snpe-onnx-to-dlc --input_network mb1-ssd-sim.onnx --output_path mb1-ssd-sim.dlc
 
-Convert image to raw using .tofile and transpose
 
 >image  =  self.transform(image) #3,300,300
images  =  image.unsqueeze(0) #1,3,300,300
images  =  images.to(self.device)
raw_img  =  images.cpu().numpy()
raw_img  =  np.transpose(raw_img,(0,2,3,1)) 
raw_img.tofile("car_transpose.raw")
>
- Use snpe-net-run to test output. Input image list is raw images line by line
>snpe-net-run --container mb1-ssd-sim.dlc --input_list image_list.txt --set_output_tensors=confidences,locations
>
- Test dlc using rundlc.py 

