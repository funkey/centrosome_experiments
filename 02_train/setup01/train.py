from funlib.learn.torch.models import UNet
import glob
import gunpowder as gp
import gunpowder.torch as gp_torch
import logging
import math
import numpy as np
import sys
import torch

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True

pos_samples = glob.glob('/nrs/funke/sheridana/centrosomes/pinky_positive_*.zarr')
neg_samples = glob.glob('/nrs/funke/sheridana/centrosomes/pinky_negative*.zarr')

print(f"Found {len(pos_samples)} pos / {len(neg_samples)} neg samples")


class Reshape(gp.BatchFilter):

    def __init__(self, array, shape):
        self.array = array
        self.shape = shape

    def process(self, batch, request):

        batch[self.array].data = np.reshape(
                batch[self.array].data,
                self.shape)

class AddCenterPoint(gp.BatchFilter):
    '''Add a single point (node in a graph) in the center of the ROI provided
    by a given array key.'''

    def __init__(self, graph_key, array_key):
        self.graph_key = graph_key
        self.array_key = array_key

    def setup(self):
        spec = gp.GraphSpec(roi=self.spec[self.array_key].roi)
        self.provides(self.graph_key, spec)
        self.center = np.array(self.spec[self.array_key].roi.get_center())

    def process(self, batch, request):
        graph = gp.Graph(
            [gp.Node(0, self.center)],
            [],
            self.spec[self.graph_key])
        batch[self.graph_key] = graph

class AddNoPoint(gp.BatchFilter):
    '''Just provides an empty graph.'''

    def __init__(self, graph_key, array_key):
        self.graph_key = graph_key
        self.array_key = array_key

    def setup(self):
        spec = gp.GraphSpec(roi=self.spec[self.array_key].roi)
        self.provides(self.graph_key, spec)

    def process(self, batch, request):
        graph = gp.Graph(
            [],
            [],
            self.spec[self.graph_key])
        batch[self.graph_key] = graph

class Convolve(torch.nn.Module):

    def __init__(
            self,
            model,
            in_channels,
            out_channels,
            kernel_size=(1,1,1)):

        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        conv = torch.nn.Conv3d

        self.conv_pass = torch.nn.Sequential(
                            conv(
                                self.in_channels,
                                self.out_channels,
                                self.kernel_size),
                            torch.nn.Sigmoid())

    def forward(self, x):

        y = self.model.forward(x)

        return self.conv_pass(y)

def train_until(max_iteration):

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factors = 6
    downsample_factors = [(1,3,3),(1,3,3),(3,3,3)]

    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors,
            constant_upsample=True)

    model = Convolve(unet, 12, 1)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-6)

    # start of gunpowder part:

    raw = gp.ArrayKey('RAW')
    points = gp.GraphKey('POINTS')
    groundtruth = gp.ArrayKey('RASTER')
    prediction = gp.ArrayKey('PRED_POINT')
    grad = gp.ArrayKey('GRADIENT')

    voxel_size = gp.Coordinate((40,4,4))

    input_shape = (96, 430, 430)
    output_shape = (60, 162, 162)

    input_size = gp.Coordinate(input_shape)*voxel_size
    output_size = gp.Coordinate(output_shape)*voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(points, output_size)
    request.add(groundtruth, output_size)
    request.add(prediction, output_size)
    request.add(grad, output_size)

    pos_sources = tuple(
        gp.ZarrSource(
            filename,
            {raw: 'volumes/raw'},
            {raw: gp.ArraySpec(interpolatable=True)}) +
        AddCenterPoint(points, raw) +
        gp.Pad(raw, None) +
        gp.RandomLocation(ensure_nonempty=points)
        for filename in pos_samples) + gp.RandomProvider()
    neg_sources = tuple(
        gp.ZarrSource(
            filename,
            {raw: 'volumes/raw'},
            {raw: gp.ArraySpec(interpolatable=True)}) +
        AddNoPoint(points, raw) +
        gp.RandomLocation()
        for filename in neg_samples) + gp.RandomProvider()

    data_sources = (pos_sources, neg_sources)
    data_sources += gp.RandomProvider(probabilities=[0.9, 0.1])
    data_sources += gp.Normalize(raw)

    train_pipeline = data_sources
    train_pipeline += gp.ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8)
    train_pipeline += gp.SimpleAugment(transpose_only=[1,2])

    train_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, \
            z_section_wise=True)
    train_pipeline += gp.RasterizePoints(
            points,
            groundtruth,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(100,100,100),
                mode='peak'))
    train_pipeline += gp.PreCache(cache_size=40, num_workers=10)

    train_pipeline += Reshape(raw, (1,1)+input_shape)
    train_pipeline += Reshape(groundtruth, (1,1)+output_shape)

    train_pipeline += gp_torch.Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'x': raw
            },
            outputs={
                0:prediction
            },
            loss_inputs={
                0:prediction,
                1:groundtruth
            },
            gradients={
                0:grad
            },
            save_every=1000,
            log_dir='log')

    train_pipeline += Reshape(raw, input_shape)
    train_pipeline += Reshape(groundtruth, output_shape)
    train_pipeline += Reshape(prediction, output_shape)
    train_pipeline += Reshape(grad, output_shape)

    train_pipeline += gp.Snapshot({
                        raw: 'volumes/raw',
                        groundtruth: 'volumes/groundtruth',
                        prediction: 'volumes/prediction',
                        grad: 'volumes/gradient'
                    },
                    every=500,
                    output_filename='test_{iteration}.hdf')
    train_pipeline += gp.PrintProfilingStats(every=10)

    with gp.build(train_pipeline):
        for i in range(max_iteration):
            train_pipeline.request_batch(request)

if __name__ == '__main__':

    iteration = 500000
    train_until(iteration)

