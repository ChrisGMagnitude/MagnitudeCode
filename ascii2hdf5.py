import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle

#projects = ['MSNT1875','MSSE209','MSSE246','MSSE247','MSSE430','MSSE469','MSSE471','MSSE562','MSSE721','MSSE1105','MSSE1394','MSSE1471','MSSE1845','MSSE1945','MSSK603','MSSK815','MSSK829','MSSK991','MSSP1306','MSSP1354','MSSP1375','MSSP1381','MSSP1392']
#projects = ['MSSK1349','MSSK1393']#,'MSSK1450','MSSK1485','MSSK1538','MSSK1591','MSSK1600','MSSK1636A','MSSK1643','MSSK1708','MSSK1743','MSSK1773','MSSK1776']
#projects = ['MSST1499','MSST1824']
#projects = ['MSST1841','MSST1921','MSSU1140','MSSU1216','MSSU1218','MSSU1270','MSSU1357','MSSU1388','MSSU1453','MSSU1526','MSSU1602','MSSU1805','MSSU1879','MSSW774']
#projects = ['MSSX1663','MSSZ1071','MSTL1402','MSTL1415A','MSTL1534']#,
projects = ['MSTM1774']
working_dir = r'F:\test\ml\cg\Classification Datasets\Extended Dataset 2\train'
#projects = ['MSSE1579','MSSK260','MSSP1489','MSSP1490','MSSP1491','MSSP1638','MSST732','MSSU552']
#working_dir = r'F:\test\ml\cg\Classification Datasets\Extended Dataset\valid'

for project in projects:
    print(project)
    points_file = os.path.join(working_dir,project+'_data_points.csv')
    points = pd.read_csv(points_file)

    ascii_file = os.path.join(working_dir,project+'.asc')

    print('Reading Ascii')
    header = {}
    with open(ascii_file) as fp:
        
        header['ncols'] = int(fp.readline().split(' ')[-1].strip())
        header['nrows'] = int(fp.readline().split(' ')[-1].strip())
        header['xllcorner'] = float(fp.readline().split(' ')[-1].strip())
        header['yllcorner'] = float(fp.readline().split(' ')[-1].strip())
        header['cellsize'] = float(fp.readline().split(' ')[-1].strip())
        header['NODATA_value'] = float(fp.readline().split(' ')[-1].strip())
    
        data = np.zeros([header['nrows'],header['ncols']])
        for i in range(header['nrows']):
            data[i,:] = fp.readline().strip().split(' ')
            
    print(header)
    if abs(header['cellsize'] - 0.25)<0.002:
        print('Normal cell size (0.25m)')
        pass
    elif abs(header['cellsize'] - 0.125)<0.002:
        print('Half cell size (0.125m)')
        data = data[::2,::2]
        header['cellsize'] = header['cellsize']*2
        header['ncols'] = int(round(header['ncols']/2))
        header['nrows'] = int(round(header['nrows']/2))
    else:
        print('Unexpected cellsize',header['cellsize'])
        print('Project',project)
        print('Aborting')
        continue
        
    print('Extracting Images')
    if not os.path.exists(os.path.join(working_dir,project)):
        os.mkdir(os.path.join(working_dir,project))
    images = []
    labels = []
    coordinates = []
    proj = []
    for i,row in points.iterrows():

        labels.append(row['Class'][:6])
        coordinates.append([row['X'],row['Y']])
        proj.append(project)

        # Pixle index of image center
        x_center = round((row['X'] - header['xllcorner'])/header['cellsize'])
        y_center = header['nrows']-round((row['Y'] - header['yllcorner'])/header['cellsize'])
        image_size = round(np.sqrt(100**2+100**2)/(header['cellsize']*2))
    
        
        image = np.ones([image_size*2,image_size*2])*-9999
    
        x_start = x_center-image_size
        x_end = x_center+image_size
        y_start = y_center-image_size
        y_end = y_center+image_size
        
        x_start_pad = abs(max([x_start,0]) - x_start)
        x_end_pad = abs(min([x_end,header['ncols']]) - x_end)
        y_start_pad = abs(max([y_start,0]) - y_start)
        y_end_pad = abs(min([y_end,header['nrows']]) - y_end)
        
        chunk = data[y_start + y_start_pad:y_end - y_end_pad,
                     x_start + x_start_pad:x_end - x_end_pad]
        
        image[y_start_pad:image_size*2-y_end_pad,x_start_pad:image_size*2-x_end_pad] = chunk
        images.append(np.float16(image))

        #image_fn = os.path.join(working_dir,project,'_'.join([str(row['Project Code']),str(row['X']),str(row['Y']),row['Class'],'.png']))
        #if os.path.exists(image_fn):
        #    continue
        # 
        #plt.imshow(-chunk,vmin = -5, vmax = 5,cmap='grey')
        #
        #plt.savefig(image_fn)
        #plt.clf()    

    print('Saving HDF5')
    images = np.array(images)
    labels = np.array(labels)
    labels = np.array([v.encode('utf-8') for v in labels])
    coordinates = np.array(coordinates)
    proj = ["{:<10}".format(a) for a in proj]
    proj = np.array([v.encode('utf-8') for v in proj])
    
    with h5py.File(working_dir+'.hdf5','a') as f:
        
        if not 'images' in f.keys():
            f.create_dataset('images', data=images, compression="lzf", chunks=(1,600,600), maxshape=(None,600,600)) 
        else:
            f["images"].resize((f["images"].shape[0] + images.shape[0]), axis = 0)
            f["images"][-images.shape[0]:] = images[:,:566,:566]
            
            
        if not 'labels' in f.keys():
            f.create_dataset('labels', data=labels, compression="lzf", chunks=True, maxshape=(None,), dtype='S6') 
        else:
            f["labels"].resize((f["labels"].shape[0] + labels.shape[0]), axis = 0)
            f["labels"][-labels.shape[0]:] = labels
    
        
        if not 'coordinates' in f.keys():
            f.create_dataset('coordinates', data=coordinates, compression="lzf", chunks=True, maxshape=(None,2)) 
        else:
            f["coordinates"].resize((f["coordinates"].shape[0] + coordinates.shape[0]), axis = 0)
            f["coordinates"][-coordinates.shape[0]:] = coordinates    
    
        
        if not 'project' in f.keys():
            f.create_dataset('project', data=proj, compression="lzf", chunks=True, maxshape=(None,), dtype='S10') 
        else:
            f["project"].resize((f["project"].shape[0] + proj.shape[0]), axis = 0)
            f["project"][-proj.shape[0]:] = proj
        
            
        print(f['images'].shape,f['labels'].shape,f['coordinates'].shape,f["project"].shape)
    