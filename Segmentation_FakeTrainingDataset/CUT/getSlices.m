function getSlices()

    x_path = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/test_x_/';
    savePath = '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/xz_slices/test_y_10d_15/';
    imgName = '10d.15';
    
    img = readStackTif(strcat(x_path, imgName, '.tif'));
    
    for sliceIx = 1:size(img, 2)
        slice = img(:, sliceIx, :);
        slice = permute(slice,[1 3 2]);

%         slice = getCellOutlines(slice);
%         slice = slice*255;

        imwrite(slice, strcat(savePath, imgName, '_', num2str(sliceIx), '.jpg'))
        
    end

%     x_directory = dir(strcat(x_path, '*.tif'));
%     
%     for imageIx = 1:length(x_directory)
%         
%         imgName = x_directory(imageIx).name;
%         
%         img = readStackTif(strcat(x_path, imgName));
%         
%         slice = img(:, :, round(end/2));
%         slice = getCellOutlines(slice);
%         imgName = strsplit(imgName, '.tif');
%         
%         imwrite(slice, strcat(savePath, imgName{1}, '.jpg'))
%         
%     end
end