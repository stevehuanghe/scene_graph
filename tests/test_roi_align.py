import unittest

import numpy as np
import torch
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.fpn.roi_align.modules.roi_align import RoIAlign
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn.functional import relu


class TestRoIAlignFunction(unittest.TestCase):

    def setUp(self):
        self.n = 3
        self.c = 8
        self.h = 10
        self.w = 12
        self.spatial_scale = 0.6
        self.sampling_ratio = 2
        self.out_c = 2
        self.aligned_height = 4
        self.aligned_width = 4

        x = np.arange(self.n * self.c * self.h * self.w, dtype=np.float32)
        x = x.reshape((self.n, self.c, self.h, self.w))
        np.random.shuffle(x)
        x = 2 * x / x.size - 1
        x = x.astype(np.float32)
        self.x = Variable(torch.Tensor(x), requires_grad=True)

        rois = np.array(
            [[0, 0, 7, 7],
             [1, 0, 5, 12],
             [0, 1, 10, 5],
             [3, 3, 4, 4]],
            dtype=np.float32
        )
        roi_indices = np.array([0, 2, 1, 0], dtype=np.int32)
        rois = np.concatenate([roi_indices[:, None], rois], axis=1)
        self.rois = Variable(torch.Tensor(rois), requires_grad=True)



    def test_backward_simple_relu(self):
        x = Variable(torch.Tensor([-1, 2, -3, 1]), requires_grad=True)
        y = relu(x)
        y.sum().backward()

        x_grad_np = x.grad.detach().cpu().data.numpy()
        assert np.array_equal(x_grad_np, [0, 1, 0, 1])

    def check_gradient_existence_gpu(self, func):
        roi_align = func(self.aligned_height, self.aligned_width, self.spatial_scale)

        x = self.x.cuda()
        rois = self.rois.cuda()

        y = roi_align(x, rois)
        loss = (y ** 2).sum()

        gx, = grad(loss, x)

        print(gx)

        assert tuple(gx.shape) == (self.n, self.c, self.h, self.w)

        del x, rois, y, loss, gx

    def check_backward_gpu(self, func):
        roi_align = func(self.aligned_height, self.aligned_width, self.spatial_scale)

        x = self.x.cuda()
        x.retain_grad()
        rois = self.rois.cuda()

        y = roi_align(x, rois)
        loss = (y ** 2).sum()
        loss.backward()

        print(x.grad)
        print(rois.grad)
        assert tuple(x.grad.shape) == (self.n, self.c, self.h, self.w)

        del x, rois, y, loss

    def test_roi_align_grad(self):
        self.check_gradient_existence_gpu(RoIAlign)

    def test_roi_align_backward(self):
        self.check_backward_gpu(RoIAlign)

    def test_roi_align_func_grad(self):
        self.check_gradient_existence_gpu(RoIAlignFunction)

    def test_roi_align_func_backward(self):
        self.check_backward_gpu(RoIAlignFunction)

    def test_save_backward_graph(self):
        roi_align = RoIAlign(self.aligned_height, self.aligned_width, self.spatial_scale)

        x = self.x.cuda()
        rois = self.rois.cuda()

        # Forward: div
        x = x / 2

        # Forward: RoIAlign
        y = roi_align(x, rois)

        # Forward: pow
        y = y ** 2

        # Forward: sum
        loss = y.sum()

        # Construct backward graph
        loss.backward()

        # Traverse the backward graph

        # Backward: sum
        sum_backward = loss.grad_fn
        assert sum_backward.__class__.__name__ == 'SumBackward0'

        pow_backward = sum_backward.next_functions[0][0]
        assert pow_backward.__class__.__name__ == 'PowBackward0'

        roi_align_backward = pow_backward.next_functions[0][0]
        assert roi_align_backward.__class__.__name__ == 'RoIAlignFunction'

        div_backward = roi_align_backward.next_functions[0][0]
        assert div_backward.__class__.__name__ == 'DivBackward0'

        del x, rois, y, loss
