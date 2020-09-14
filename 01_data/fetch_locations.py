import cloudvolume
import daisy
import json
import logging
import numpy as np
import os
import sys

logging.basicConfig(level=logging.INFO)

def shrink_bbox_mip(bbox):

    bb_center = bbox.center()
    w,h,d = bbox.size3()

    minpt = bb_center - np.array([w//4,h//4,d//2])

    return list(cloudvolume.Bbox.from_delta(minpt, [w//2, h//2, d]).minpt)

def convert_coords(
        coords,
        voxel_size,
        method='world_to_vox',
        flip_coords=False):

    if method=='world_to_vox':
        if flip_coords:
            return [int(i/j) for i,j in zip(coords, voxel_size)][::-1]
        else:
            return [int(i/j) for i,j in zip(coords, voxel_size)]
    else:
        if flip_coords:
            return [int(i*j) for i,j in zip(coords, voxel_size)][::-1]
        else:
            return [int(i*j) for i,j in zip(coords, voxel_size)]

def fetch_in_block(
        block,
        voxel_size,
        raw_data,
        out_ds):

    logging.info('Fetching seg in block %s' %block.read_roi)

    voxel_size = list(voxel_size)

    block_start = list(block.write_roi.get_begin())
    block_end = list(block.write_roi.get_end())

    block_start = convert_coords(block_start,voxel_size)
    block_end = convert_coords(block_end,voxel_size)

    z_start, z_end = block_start[0], block_end[0]
    y_start, y_end = block_start[1], block_end[1]
    x_start, x_end = block_start[2], block_end[2]

    raw = raw_data[x_start:x_end, y_start:y_end, z_start:z_end]

    raw = np.array(np.transpose(raw[...,0],[2,1,0]))

    print(raw)

    out_ds[block.write_roi] = raw

def fetch(
        in_vol,
        voxel_size,
        roi_offset,
        roi_shape,
        out_file,
        out_ds,
        num_workers):

    total_roi = daisy.Roi((roi_offset), (roi_shape))

    read_roi = daisy.Roi((0,)*3, (4800,1280,1280))
    write_roi = read_roi

    logging.info('Creating out dataset...')

    raw_out = daisy.prepare_ds(
            out_file,
            out_ds,
            total_roi,
            voxel_size,
            dtype=np.uint8,
            write_roi=write_roi)

    logging.info('Writing to dataset...')

    daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: fetch_in_block(
                b,
                voxel_size,
                in_vol,
                raw_out),
            fit='shrink',
            num_workers=num_workers) 
if __name__ == '__main__':

    # vol = "https://storage.googleapis.com/a-j-cilium-project-ngl/neuroglancer_cutouts/pinky100"
    vol = "https://storage.googleapis.com/a-j-cilium-project-ngl/neuroglancer_cutouts/basil"

    in_vol = cloudvolume.CloudVolume(
            vol,
            bounded=True,
            progress=True,
            fill_missing=True)

    voxel_size = daisy.Coordinate(
                    in_vol.info['scales'][0]['resolution'][::-1]
                )

    with open(sys.argv[1], 'r') as f:
        points = json.load(f)

    out_path = sys.argv[2]

    for p in points:
        minpt = shrink_bbox_mip(cloudvolume.Bbox.from_dict(p))[::-1]

        minpt[1] = minpt[1]*2
        minpt[2] = minpt[2]*2

        roi_offset = daisy.Coordinate([i*j for i,j in zip(minpt,voxel_size)])
        roi_shape = daisy.Coordinate((4800,3680,3680))

        out_file = os.path.join(out_path, 'location_%d.zarr' % (points.index(p)+1))
        out_ds = 'volumes/raw'

        try:
            fetch(
                in_vol,
                voxel_size,
                roi_offset,
                roi_shape,
                out_file,
                out_ds,
                num_workers=8)
        except:
            pass

