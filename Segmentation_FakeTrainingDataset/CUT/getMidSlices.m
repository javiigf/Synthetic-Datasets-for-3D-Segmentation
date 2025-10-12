function getMidSlices()

    x_path = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/y/';
    savePath = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/y_midSlices/';

    x_directory = dir(strcat(x_path, '*.tif'));
    
    for imageIx = 1:length(x_directory)
        
        imgName = x_directory(imageIx).name;
        
        img = readStackTif(strcat(x_path, imgName));
        
        slice = img(:, round(end/2), :);
        slice = permute(slice,[1 3 2]);
        
        slice = getCellOutlines(slice);
        imgName = strsplit(imgName, '.tif');
        
        imwrite(slice, strcat(savePath, imgName{1}, '.jpg'))
        
    end
end