import numpy as np
import onnxruntime as rt


class SeqGraphUin2UinInfer:
    def __init__(self, data, args_dict):
        self.infer_dict = args_dict
        self.data = data
        self.sess = rt.InferenceSession(self.infer_dict['model_onnx_path'])

    def infer(self):
        print("self.sess.get_inputs()", self.sess.get_inputs())
        seqs_x = self.data.test_seq_token_id.values.astype(int)
        input_x = self.data.test_input_feat.astype(np.float32)
        target_x = self.data.test_target_feat.astype(np.float32)

        input_names = [input.name for input in self.sess.get_inputs()]

        input_dict = {
            input_names[0]: seqs_x,
            input_names[1]: input_x,
            input_names[2]: target_x
        }

        print("self.sess.get_outputs()", self.sess.get_outputs())

        outputs = self.sess.run(None, input_dict)

        print("outputs", outputs)

        mutil_cls_pred = outputs[0]

        print("Prediction:", mutil_cls_pred)
