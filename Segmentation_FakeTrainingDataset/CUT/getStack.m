function getStack()

    x_path = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/fake_B_10d_15/';
    savePath = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/fake_B_stack/';

    x_directory = dir(strcat(x_path, '*.png'));
    
    for imageIx = 1:length(x_directory)
        
        imgName = x_directory(imageIx).name;
        
        img = imread(strcat(x_path, imgName));
        
        img = rgb2gray(img);
%         img = im2gray(img);

        disp(imageIx)
        
        stackImg(:, imageIx, :) = img;

        
    end
    imgName = '10d_15_fake';
    
    stackImg = imresize3(stackImg, [256, 256, 126]);
    writeStackTif(stackImg, strcat(savePath, imgName, '.tif'))

end