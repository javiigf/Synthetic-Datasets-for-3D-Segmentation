function resizeAll(path1, path2, minSize)
    minSize = [180,180,64];
    path1 = '/home/jesus/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main/dataset/dataset_20250625_embryo_gaussian/trainB/';
    path2 = '/home/jesus/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main/dataset/dataset_20250625_embryo_gaussian/trainB_resized/';
    
    imgs_dir = dir(strcat(path1, '*', '.tif'));
    
    for n_file = 1:length(imgs_dir)
        img = readStackTif(strcat(path1, imgs_dir(n_file).name));
        imgSize = size(img);
        
        imgName = imgs_dir(n_file).name;
        imgName = strsplit(imgName, '.tif');
        imgName = imgName{1};
        
       newImg = imresize3(img, minSize, 'nearest');
       writeStackTif(newImg/255, strcat(path2, imgName, '_resized.tif'));

    end

end