
# ================ DESCRIPTION ==============================================================================

# This file applies a LoG detector on fluorescent images (.vsi or .tiff, both works) in order to detect cells.
# More specifically, it retrieves the background of the image at its first frame to compute the SNR (signal
# over noise ratio) then applies a smoothing (parameter sigma) then Laplacian of gaussians to identify
# the contours of the spots with a value of threshold (parameter s) to select them. Hence, after
# these 3 steps, we have detected "blobs". Now we want to retrieve the coordinates of these blobs.
# For that, an algorithm of search for local maxima is applied onto the blobs. Finally, these coordinates
# are saved under a dictionnary of dataframes. See commands below to acces the data in it.

# Commands dataframes:
    
#   to open with pandas, use:
#       store=pandas.HDFStore(filename,'r')
#       p=pandas.read_hdf(store,key=num_frame)
#       store.close()
#
#   to open with h5py:
#       f=h5py.File(filename,'r')
#   to get the information saved in the attributes:
#       f.attrs.keys()
#   and then:
#       f.attrs[name_atribute]
#   to get the background:
#       f['background']['im'][:] as an array -> plt.imshow() to see it
#   to get one of the dataframes:
#       
#
# Command in the terminal to launch this code:
#
#   python Detection_algorithm.py "path_*.extension"  "path_save.hdf5"
#

# =============== REQUIRED PACKAGES =========================================================================================

import javabridge # To use bioformats 
import bioformats # to open the images
import sep # to determine the background
import Find_Local_Maxima as findMax # python file to determine blobs and local maxima
import h5py # to have compacted dataframes
import pandas # to use dataframes
import sys,os # to access our images and use the terminal
import glob # finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
from tqdm import tqdm # to get a processing bar in the terminal
from scipy.spatial import KDTree

import numpy as np

# =============== SETTINGS =========================================================================================

# We need to specify the channel used in the images. If we have bright field images with 
# fluorescence microscopy images, we need to put it at 1 !
channel=0
# To identify the background, we define a window that crosses the image. Should not be too small to avoid
# local effects
bw=256
# The smoothing
sigma=3.5
# Threshold of the Laplacian
s=0.0375
# Minimum of pixels in a spot to be considered as a spot and not noise
ccmin=30
# We can also specify a maximum number of pixels to define a spot if we wish to, otherwise set to None
ccmax=None
# If we need to eliminate repeat count in a defined distance
distance_min=12

# =============== COMMANDS =========================================================================================

# In this part, messages are printed in the terminal to verify that we will apply the algorithm to the right images
# and save the resulting file under the right name/directory

# Check all the images that corresponds to the criteria given for the pathname in the terminal
#vsis=glob.glob(sys.argv[1])
vsis = glob.glob("./20251003 f98 fucci-1 last line/_Image_D1_01_F98_Fucci__/stack1/*.ets")
print(vsis)
# retrieve the expected output filename
#fout=sys.argv[2]
fout = "./output_file_D1_01_0.0375.hdf5"
# Then check that all images do verify the criteria
#dirin=os.path.dirname(sys.argv[1])
dirin=os.path.dirname("./20251003 f98 fucci-1 last line/_Image_D1_01_F98_Fucci__/stack1/*.ets")
print("input dir={}".format(dirin))
for vsi in vsis:
    base=os.path.basename(vsi)
    print(base)
#    assert base.split("_")[-2]=="Image", "wrong vsi={}".format(base)
print("output file will be ={}".format(fout))
ans=input("is this OK? (y/n) ")
# To allow us to confirm or not 
if ans != "y":
    sys.exit()


#Then we start java to access our images
javabridge.start_vm(class_path=bioformats.JARS)
# we create the output file to write in it
f=h5py.File(fout,'w')
 
# we specify the metadata in our output file
f.attrs['channel']=channel
f.attrs['sigma']=sigma
f.attrs['seuil']=s
f.attrs['ccmin']=ccmin
if ccmax is not None:
    f.attrs['ccmax']=ccmax
# for pnadas df
store=pandas.HDFStore(fout,'a')

# Then we loop on the images 
for ifile,fin in enumerate(vsis):
    # determine the image number from file name
    b=os.path.basename(fin)
#    imnum=int(b.split(".")[0].split("Image_")[1])
#    imnum = int(b.split(".")[0].split("_")[-1].replace("d", "").replace("h", "").replace("m", ""))
#    imnum = int(b.split("_")[2])
#    imnum = f"{b.split('_')[0]}_{b.split('_')[1]}_{b.split('_')[2]}_{b.split('_')[3]}"
    imnum = f"{b.split('_')[0]}"


    # open the image with bioformats
    ome=bioformats.OMEXML(bioformats.get_omexml_metadata(fin))
    print(ome.image().AcquisitionDate)
    # Retrieve properties of the image
    nt=ome.image().Pixels.SizeT
    Nx=ome.image().Pixels.SizeX
    Ny=ome.image().Pixels.SizeY
    nchan=ome.image().Pixels.channel_count
    # read the image 
    reader=bioformats.ImageReader(fin)


    # HDF output (how information will be organized in the out file)
    print("><"*100)
    print(r"{} ({}/{})".format(fin,ifile,len(vsis)))
    print(r" ->{} frames. image size=({}x{}) with {} channels: using channel={}".format(nt,Nx,Ny,nchan,channel))
    
    # we loop on the frames
    for ip in tqdm(range(nt),desc="Processing"):
        yy=reader.read(c=channel,t=ip)
        if yy.ndim == 3 and yy.shape[2] == 3:  # examiner si c'est un RGB image
    # seperate the 3 channels
            yy = yy[:, :, 0]   # R channel
            data=yy.reshape(Ny,Nx)
            data = np.ascontiguousarray(data)
        
        else:
            data=yy.reshape(Ny,Nx)
        # compute background if we are at the first frame and save it in the output file
        if ip==0:
            bkg=sep.Background(data,bw=bw,bh=bw)
            key="Image{}/background".format(imnum)
            g = f.create_group(key)
            ds=g.create_dataset("image",data=bkg.back(),compression='gzip')
            g.attrs['rms']=bkg.globalrms
            g.attrs['bw']=bw

        # work on SNR
        snr=(data-bkg.back())/bkg.globalrms
        # compute LoG and threshold to obtain blobs
        blobs=findMax.getBlobs(snr,s=s,ccmin=ccmin,sigma=sigma,ccmax=None)
        # get local maxima for each blob: output is a dataframe
        pos=findMax.findMax(blobs,data.shape)
        
        
        #pos = findMax.filter_coordinates(pos, distance_min)
        #pos = pandas.DataFrame(pos, columns=['x', 'y'])
        # write df (careful, here floats)
        key="Image{}/frame{}".format(imnum,ip)
        pos.to_hdf(store,key=key)

# we close the files created
store.close()
f.close()
# and java
javabridge.kill_vm()
