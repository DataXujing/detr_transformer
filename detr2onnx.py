# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import unittest

import torch
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False, do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        onnx_path = "detr.onnx"

        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io,
            input_names=input_names, output_names=output_names,export_params=True,training=False)
        torch.onnx.export(model, inputs_list[0], onnx_path,
            input_names=input_names, output_names=output_names,export_params=True,training=False)
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def test_model_onnx_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes"],
            tolerate_small_mismatch=True,
        )


if __name__ == '__main__':


    detr = detr_resnet50(pretrained=False,num_classes=3+1).eval()  # <------这里类别需要+1
    state_dict =  torch.load('./outputs/checkpoint.pth')   # <-----------修改加载模型的路径
    detr.load_state_dict(state_dict["model"])
    

    dummy_image = [torch.ones(1, 3, 800, 800) ]

    onnx_export = ONNXExporterTester()
    onnx_export.run_model(detr, dummy_image,input_names=['inputs'],
        output_names=["pred_logits", "pred_boxes"],tolerate_small_mismatch=True)
    # https://colab.research.google.com/drive/18UBY-mY9tuw22I4RdjoTua_JfpTTBcE7?usp=sharing
    # torch.onnx.export(detr, dummy_image, "detr.onnx",
    #     input_names=['inputs'], output_names=["pred_logits", "pred_boxes"])

    
