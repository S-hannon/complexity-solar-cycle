from __future__ import division
import os
import fnmatch
import shutil
import matplotlib as mpl
import sunpy.map as smap
from PIL import Image
from datetime import datetime, timedelta
import hi_processing.images as hip


########### NOTE
# data is stored in data_loc\STEREO_HI
# images are stored in data_loc\STEREO_HI\Images\tag\img_type\craft


class STEREOHI:
    def __init__(self, data_loc):
        self.hi_data_2007 = r'D:\STEREO\ares.nrl.navy.mil\lz'
        self.hi_data_2014 = r'E:'
        self.data_loc = os.path.join(data_loc, 'STEREO_HI')
        if not os.path.exists(self.data_loc):
            os.mkdir(self.data_loc)
        
    
    def __get_correct_drive(self, time):
        """
        time: datetime instance
        returns path to correct hard drive
        """
        if time.year < 2014:
            return self.hi_data_2007
        else:
            return self.hi_data_2014
    
    
    def save_fits_files(self, start_time, end_time, craft='sta', camera='hi1',
                        background_type=1):
        """function to download .fits files from hard drive.
        """
        hi_files = hip.find_hi_files(start_time, end_time, craft=craft,
                                     camera=camera,
                                     background_type=background_type)
        for i in range(len(hi_files)):
            # using first file, make new folders
            path_end_plus_file_name = hi_files[i].split(self.hi_data2007)[1]
            parts = path_end_plus_file_name.split('\\')
            path_end = ('\\').join(parts[1:len(parts)-1])            
            # make folder to store data
            if not os.path.exists(os.path.join(self.data_loc, path_end)):
                os.makedirs(os.path.join(self.data_loc, path_end))
            file_name = parts[len(parts)-1]
            shutil.copy2(hi_files[i], os.path.join(self.data_loc, path_end,
                                                   file_name))
        
    
    def get_hi_map(self, craft, time):
        """Retrieves hi_map from hard drive for a given CME, using the time the
        CME was first observed in HI, given in the HELCATS ID for the CME.
        """
        day_string = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2)
        time_string = str(time.hour).zfill(2) + str(time.minute).zfill(2)
        drive_path = self.__get_correct_drive(time)
        # Find the folder on the correct day
        hi_folder = os.path.join(drive_path, "L2_1_25", str(craft[2]),
                                 'img\hi_1', day_string)
        # find correct file from folder
        print(hi_folder)
        print(day_string + "_" + time_string)
        for filename in os.listdir(hi_folder):
            if filename.startswith(day_string + "_" + time_string):
                hi_map = smap.Map(os.path.join(hi_folder, filename))
        return hi_map
    
    
    ###########################################################################
    # for making images
    def __check_inputs(self, img_tag, craft, tag, img_type, time1=None,
                       time2=None):
        if craft not in ['sta', 'stb']:
            raise ValueError(str(craft) + " must be sta or stb")
        if img_type not in ['diff', 'norm']:
            raise ValueError(str(img_type) + " must be diff or norm")
        for in_str in [tag, img_tag]:
            if not isinstance(in_str, str):
                raise ValueError(str(in_str) + "must be str")
        for time in [time1, time2]:
            if time != None:
                if not isinstance(time, datetime):
                    raise ValueError("time must be datetime instance")  
                    
                    
    def __find_img_file(self, time, craft, camera='hi1', background_type=1):
        # TODO Check date input within range
        # Search hard drive for matching HI image
        hi_files = hip.find_hi_files(time - timedelta(hours=2), time,
                                     craft=craft, camera=camera,
                                     background_type=background_type)
        if len(hi_files) == 0:
            # Try 5 mins later, should avoid errors with the seconds being wrong
            # this is okay as cadence ~40 mins
            hi_files = hip.find_hi_files(time - timedelta(hours=2),
                                         time + timedelta(minutes=5),
                                         craft=craft, camera=camera,
                                         background_type=background_type)     
        fc = None
        fp = None
        if len(hi_files) > 1:
            # Loop over the hi_files, make image
            fc = hi_files[len(hi_files)-1]  # current frame files, last in list
            fp = hi_files[len(hi_files)-2]  # previous frame files, 2nd last
        return fc, fp
    
    
    def __find_img_files(self, start_time, end_time, craft, camera='hi1',
                         background_type=1):
        # Search hard drive for matching HI image
        hi_files = hip.find_hi_files(start_time, end_time, craft,
                                     camera=camera,
                                     background_type=background_type)
        fc_list = hi_files[1:]
        fp_list = hi_files[0:-1]
        return fc_list, fp_list


    def __make_norm_img(self, fc):
        hi_map = hip.get_image_plain(fc, star_suppress=False)
        plain_normalise = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
        img = mpl.cm.gray(plain_normalise(hi_map.data), bytes=True)
        return img, hi_map

                   
    def __make_diff_img(self, fc, fp):
        hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
        diff_normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
        img = mpl.cm.gray(diff_normalise(hi_map.data), bytes=True)
        img = Image.fromarray(img)
        return img, hi_map


    def __save_img(self, img, hi_map, img_tag, craft, tag, img_type, ):
        name = "_".join([img_tag, craft,
                         hi_map.date.strftime('%Y%m%d_%H%M%S')]) + '.jpg'
        out_path = os.path.join(self.data_loc, 'Images', tag, img_type,
                                craft, name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        img = img.convert(mode='RGB')
        img.save(out_path)
    
    
    def make_img(self, img_tag, time, craft, tag, img_type,
                 camera='hi1', background_type=1):
        """Makes image for the required CME using the .FTS files
        on the hard-drives.
        time: datetime
        craft: str, "sta" or "stb"
        tag: str, folder to save all images in, e.g. "POPFSS"
        img_type: str, "norm" or "diff"
        img_tag: str, to save at start of image name, e.g. a helcats cme name
        camera: str, "hi1" or "hi2"
        background_type: int, 1 or 11 day
        """
        self.__check_inputs(img_tag, craft, tag, img_type, time1=time)
        fc, fp = self.__find_img_file(time, craft, camera=camera,
                                      background_type=background_type)
        if img_type == 'diff':
            img, hi_map = self.__make_diff_img(fc, fp)
        else:
            img, hi_map = self.__make_norm_img(fc)
        self.__save_img(img, hi_map, img_tag, craft, tag, img_type)

            
    def make_imgs(self, img_tag, start_time, end_time, craft, tag, img_type,
                  camera='hi1', background_type=1):
        """Makes image for the required CME using the .FTS files
        on the hard-drives.
        start_time, end_time: datetime
        craft: str, "sta" or "stb"
        tag: str, folder to save all images in, e.g. "POPFSS"
        img_type: str, "norm" or "diff"
        img_tag: str, to save at start of image name, e.g. a helcats cme name
        camera: str, "hi1" or "hi2"
        background_type: int, 1 or 11 day
        """
        self.check_inputs(img_tag, craft, tag, img_type,
                          time1=start_time, time2=end_time)
        fc_list, fp_list = self.__find_img_files(start_time, end_time, craft,
                                                 camera=camera,
                                                 background_type=background_type)
        for n in range(len(fc_list)):
            if img_type == 'diff':
                img, hi_map = self.__make_diff_img(fc_list[n], fp_list[n])
            else:
                img, hi_map = self.__make_norm_img(fc_list[n])
            self.__save_img(img, hi_map, craft, tag, img_type, img_tag=img_tag)
        

    ###########################################################################  
    def load_img(self, img_tag, craft, tag, img_type):
        """Loads the image for the required CME from the folder specified in the
        project directiories.
        """
        if not isinstance(img_tag, datetime):
            if not isinstance(img_tag, str):
                raise ValueError('img_tag must be datetime or str')
        else:
            # convert datetime to string
            img_tag = img_tag.strftime("%Y%m%d_%H%M%S")
        # Loop over files in the folder
        c = 0
        file_list = os.listdir(os.path.join(self.data_loc, 'Images', tag,
                                            img_type, craft))
        while c < len(file_list):
            if fnmatch.fnmatch(file_list[c], '*' + img_tag + '*'):
                file_name = file_list[c]
                break
            c = c + 1   
        img = Image.open(os.path.join(self.data_loc, 'Images', tag,
                                      img_type, craft, file_name))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    