import torch

class XingLoss(torch.nn.Module):

    def area(self, a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


    def triangle_area(self, A, B, C):
        out = (C - A).flip([-1]) * (B - A)
        out = out[..., 1] - out[..., 0]
        return out


    def compute_sine_theta(self, s1, s2):  # s1 and s2 are two segments to be used
        # s1, s2 (2, 2)
        v1 = s1[1, :] - s1[0, :]
        v2 = s2[1, :] - s2[0, :]
        # print(v1, v2)
        k = torch.norm(v1) * torch.norm(v2)
        sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / k
        return sine_theta

    def forward(self, x_list, scale=1):
        # x[ npoints,2]
        loss = 0.
        # print(len(x_list))
        for x in x_list:
            # print(x)
            seg_loss = 0.
            N = x.size()[0]
            x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
            segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
            assert N % 4 == 0, 'The segment number is not correct!'
            # NOTE: changed N % 3 on N % 4
            segment_num = int(N / 4)
            for i in range(segment_num):
                cs1 = segments[i * 3, :, :]  # start control segs
                cs2 = segments[i * 3 + 1, :, :]  # middle control segs
                cs3 = segments[i * 3 + 2, :, :]  # end control segs
                # print('the direction of the vectors:')
                # print(compute_sine_theta(cs1, cs2))
                direct = (self.compute_sine_theta(cs1, cs2) >= 0).float()
                opst = 1 - direct  # another direction
                sina = self.compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
                seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
                # print(direct, opst, sina)
            seg_loss /= segment_num

            templ = seg_loss
            loss += templ * scale  # area_loss * scale
        loss /= (len(x_list))
        return loss
