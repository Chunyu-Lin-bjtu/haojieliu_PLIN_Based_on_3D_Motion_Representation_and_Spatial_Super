# Usage: python xxx.py path_to_pc1/pc2/output/epe3d/path_list [pc2]

import numpy as np
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle

SCALE_FACTOR = 0.05
MODE = 'sphere'
DRAW_LINE = True

if '-h' in ' '.join(sys.argv):
	print('Usage: python3 visu_new.py VISU_PATH')
	sys.exit(0)

# visu_path = sys.argv[1]
# visu_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/test/ours_KITTI_train_3000_35m/val_output'
# visu_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/test/ours_KITTI_8192_35m/visu_ours_KITTI_8192_35m'
# visu_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/test/ours_KITTI_train_3000_35m/visu_ours_KITTI_train_3000_35m'
# visu_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/test/ours_KITTI_train_3000_35m/val_output_remove_road'
visu_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/data/val_output_remove_road'

# all_epe3d = np.load(osp.join(visu_path, 'epe3d_per_frame.npy'))

# path_list = None
# if osp.exists(osp.join(visu_path, 'sample_path_list.pickle')):
# 	with open(osp.join(visu_path, 'sample_path_list.pickle'), 'rb') as fd:
# 		path_list = pickle.load(fd)
		
# for index in range(len(path_list)):
for index in range(20):
	pc1 = np.load(osp.join(visu_path, 'pc1_'+str(index)+'.npy')).squeeze()
	pc2 = np.load(osp.join(visu_path, 'pc2_'+str(index)+'.npy')).squeeze()
	# sf = np.load(osp.join(visu_path,  'sf_'+str(index)+'.npy')).squeeze()
	output = np.load(osp.join(visu_path, 'output_'+str(index)+'.npy')).squeeze()

	# pc = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/data/kitti_rgb/val/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin'
	# pc_temp = np.load(pc)
	
	if pc1.shape[1] != 3:
		pc1 = pc1.T
		pc2 = pc2.T
		# sf = sf.T
		output = output.T
	
	# gt = pc1 + sf
	# pred = pc1 + output
	pred = pc1 + output
	
	print('pc1, pc2, gt, pred', pc1.shape, pc2.shape, pred.shape)


	fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=(1,1,1), engine=None, size=(1600, 1000))
	
	if True: #len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
		mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
		# mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # blue
	
	# if len(sys.argv) >= 4 and sys.argv[3] == 'pc2':
	mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # red

	# mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(1,0,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # red
	mlab.points3d(pred[:, 0], pred[:,1], pred[:,2], color=(0,1,0), scale_factor=SCALE_FACTOR, figure=fig, mode=MODE) # green
	
	# epe3d = all_epe3d[index]
	# print(epe3d)
	# path = path_list[index]
	# print(path, epe3d)	
	
	# # DRAW LINE
	# if DRAW_LINE:
	# 	N = 2
	# 	x = list()
	# 	y = list()
	# 	z = list()
	# 	connections = list()

	# 	inner_index = 0
	# 	for i in range(gt.shape[0]):
	# 		x.append(gt[i, 0])
	# 		x.append(pred[i, 0])
	# 		y.append(gt[i, 1])
	# 		y.append(pred[i, 1])
	# 		z.append(gt[i, 2])
	# 		z.append(pred[i, 2])

	# 		connections.append(np.vstack(
	# 			[np.arange(inner_index,   inner_index + N - 1.5),
	# 			np.arange(inner_index + 1,inner_index + N - 0.5)]
	# 		).T)
	# 		inner_index += N

	# 	x = np.hstack(x)
	# 	y = np.hstack(y)
	# 	z = np.hstack(z)

	# 	connections = np.vstack(connections)

	# 	src = mlab.pipeline.scalar_scatter(x, y, z)

	# 	src.mlab_source.dataset.lines = connections
	# 	src.update()
		
	# 	lines= mlab.pipeline.tube(src, tube_radius=0.005, tube_sides=6)
	# 	mlab.pipeline.surface(lines, line_width=2, opacity=.4, color=(1,1,0))
	# # DRAW LINE END

	
	mlab.view(90, # azimuth
	         150, # elevation
			 50, # distance
			 [0, -1.4, 18], # focalpoint
			 roll=0)

	mlab.orientation_axes()

	mlab.show()
