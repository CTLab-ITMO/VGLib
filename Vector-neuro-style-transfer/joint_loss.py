from math import sqrt

import bezier
import numpy as np
import torch

class JointLoss(torch.nn.Module):

    def get_coordinates(self, t, points):
        curve = bezier.Curve(points, degree=3)
        return curve.evaluate(t)


    def is_local_max(self, t, curve):
        eps = 0.1
        return curve.evaluate(t)[1][0] > curve.evaluate(t + eps)[1][0] and curve.evaluate(t)[1][0] > \
            curve.evaluate(t-eps)[1][0]


    def is_local_min(self, t, curve):
        eps = 0.1
        return curve.evaluate(t)[1][0] < curve.evaluate(t + eps)[1][0] and curve.evaluate(t)[1][0] < \
            curve.evaluate(t - eps)[1][0]


    class AnchorPoints:
        a0, b0 = 0, 0
        a1, b1 = 0, 0
        a2, b2 = 0, 0
        a3, b3 = 0, 0

        def __init__(self, points: np.ndarray):
            self.a0, self.b0 = points[0][0], points[0][1]
            self.a1, self.b1 = points[1][0], points[1][1]
            self.a2, self.b2 = points[2][0], points[2][1]
            self.a3, self.b3 = points[3][0], points[3][1]

        def swap_points(self):
            ap = self
            ap.a0, ap.b0 = self.b0, self.a0
            ap.a1, ap.b1 = self.b1, self.a1
            ap.a2, ap.b2 = self.b2, self.a2
            ap.a3, ap.b3 = self.b3, self.a3
            return ap

        def generate_curve(self):
            return bezier.Curve(np.asfortranarray([
                [self.a0, self.a1, self.a2, self.a3],
                [self.b0, self.b1, self.b2, self.b3]
            ]), degree=3)


    def find_joint(self, points, anchor_points):
        curve = anchor_points.generate_curve()
        a0 = anchor_points.a0
        a1 = anchor_points.a1
        a2 = anchor_points.a2
        a3 = anchor_points.a3
        k = a2 - 2 * a1 + a0
        a = a3 - 3 * a2 + 3 * a1 - a0
        c = a1 - a0
        if a == 0:
            swapped_points = np.swapaxes(points, 0, 1)
            swapped_ap = anchor_points.swap_points()
            swapped_point, t_res = self.find_joint(points=swapped_points, anchor_points=swapped_ap)
            if t_res == -1:
              return np.ndarray(shape=(0, 0)), -1
            return np.array([swapped_point[1], swapped_point[0]]), t_res
        d1 = k * k - a * c
        if d1 < 0:
            return np.ndarray(shape=(0, 0)), -1
        t1 = (-1 * k + sqrt(d1)) / a
        t2 = (-1 * k - sqrt(d1)) / a
        if 1 >= t1 >= 0:
            if self.is_local_max(t1, curve) or self.is_local_min(t1, curve):
                return curve.evaluate(t1), t1
        if 1 >= t2 >= 0:
            if self.is_local_max(t2, curve) or self.is_local_min(t2, curve):
                return curve.evaluate(t2), t2
        return np.ndarray(shape=(0, 0)), -1


    def joint_loss(self, points: torch.Tensor, eps=0.1):
      # print(points)
      np_points = points.detach().numpy()
      # print(np_points)
      ap = self.AnchorPoints(np_points)
      joint, t_joint = self.find_joint(np_points, ap)

      def der_x(t): return 3*ap.a3*t*t - 3*ap.a2*(3*t*t - 2*t) + 3*ap.a1*(3*t*t - 4*t + 1) - 3*ap.a0*(t*t - 2*t + 1)
      def der_y(t): return 3*ap.b3*t*t - 3*ap.b2*(3*t*t - 2*t) + 3*ap.b1*(3*t*t - 4*t + 1) - 3*ap.b0*(t*t - 2*t + 1)
      def double_der_x(t): return 6*t*(ap.a3 - 3*ap.a2 + 3*ap.a1 - ap.a0) + 6*(ap.a2 - 2*ap.a1 + ap.a0)
      def double_der_y(t): return 6*t*(ap.b3 - 3*ap.b2 + 3*ap.b1 - ap.b0) + 6*(ap.b2 - 2*ap.b1 + ap.b0)

      def k(t):
          x_ap = der_x(t)
          y_ap = der_y(t)
          x_ap_ap = double_der_x(t)
          y_ap_ap = double_der_y(t)
          norm = sqrt(x_ap * x_ap + y_ap * y_ap)
          if norm == 0: return 0.
          return (x_ap * y_ap_ap - x_ap_ap * y_ap) / (norm * norm * norm)

      k_left = k(t_joint - eps)
      k_right = k(t_joint + eps)
      return sqrt(abs(k_left) + abs(k_right))

    def forward(self, x_list, eps=0.1):
      loss = 0.
      # print(len(x_list))
      for x in x_list:
        # print(x)
        # area_loss * scale
        loss += self.joint_loss(x, eps)
      return torch.tensor(loss)
