import os
import numpy as np
import torch
import torch.onnx
from upcunet_v3 import RealWaifuUpScaler
import torch.jit
import torch.export
import coremltools as ct
import numpy as np

# upsampler = RealWaifuUpScaler(scale=2, 
                            #   weight_path="/Users/apleynes/Documents/dev/Swift-SRGAN/realcugan/Real-CUGAN/weights_v3/up2x-latest-no-denoise.pth", half=False, device="cpu")
# weight_path = "/Users/apleynes/Documents/dev/Swift-SRGAN/realcugan/Real-CUGAN/weights_v3/up2x-latest-no-denoise.pth"
weight_path = "/Users/apleynes/Documents/dev/Swift-SRGAN/realcugan/Real-CUGAN/weights_v3/up2x-latest-denoise1x.pth"
upsampler = RealWaifuUpScaler(scale=2, 
                              weight_path=weight_path, half=False, device="cpu")


# upsampler.load_state_dict(torch.load(os.path.join("checkpoints", "swift_srgan_2x.pth.tar"), map_location="cpu")['model'])
# upsampler.load_state_dict(torch.load(os.path.join("checkpoints", "netG_2x_epoch100.pth.tar"), map_location="cpu")['model'])
# upsampler.eval()
# upsampler = upsampler.half()
pro = upsampler.pro
print(pro)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        return self.model(x, tile_mode=0, cache_mode=0, alpha=1, pro=pro)

model = ModelWrapper(upsampler.model)
model.eval()
traced_upsampler = torch.jit.trace(model, torch.randn(1, 3, 240, 320), strict=False)
# exported = torch.export.export(ModelWrapper(upsampler.model), args=(torch.randn(1, 3, 256, 256),))

# TODO: Add shape range for input spec
# Save for inspection
torch.jit.save(traced_upsampler, "traced_upsampler.pt")


# shape_range = ct.Shape(shape=(1, 
                            #   3, 
                            #   ct.RangeDim(lower_bound=240, upper_bound=1920, default=240), 
                            #   ct.RangeDim(lower_bound=320, upper_bound=2560, default=320)))
shape_range = (1, 3, 240, 320)
# shape_range = ct.EnumeratedShapes(shapes=[(1, 3, 240, 320), (1, 3, 480, 640), (1, 3, 960, 1280)])
input_type = ct.ImageType(shape=shape_range, scale=1/255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB)
# output_type = ct.ImageType(scale=1.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB)

mlmodel = ct.convert(traced_upsampler, inputs=[input_type], #outputs=[output_type],
                     minimum_deployment_target=ct.target.macOS15,
                     compute_precision=ct.precision.FLOAT16,
                     convert_to='mlprogram',
                     debug=False)
# mlmodel = ct.convert(exported, inputs=[ct.TensorType(shape=(1, 3, 256, 256), dtype=np.float16)], 
                    #  minimum_deployment_target=ct.target.macOS15, debug=True)
# mlmodel.save("up2x-latest-no-denoise.mlpackage")
output_filename = os.path.basename(weight_path).replace('.pth', '.mlpackage')
mlmodel.save(output_filename)

# Bugs on doing flexible shapes. So just need to export different models for different shapes.
# Including the jit trace operation due to how U-net works
# Relevant github issues:
# https://github.com/apple/coremltools/issues/2037
# https://github.com/apple/coremltools/issues/2159
# https://github.com/apple/coremltools/issues/2160