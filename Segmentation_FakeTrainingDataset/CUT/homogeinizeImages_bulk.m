rgStackPath = strcat('/media/pedro/6TB/jesus/eggChambers/batch_20221110/S5-6_ResilleGFP_27.10.22_Crop from 1024X1024_40X/');
rgStackPath2 = strcat('/media/pedro/6TB/jesus/eggChambers/batch_20221110/S5-6_ResilleGFP_27.10.22_Crop from 1024X1024_40X/');

savePath = strcat('/media/pedro/6TB/jesus/eggChambers/batch_20221110/S5-6_ResilleGFP_27.10.22_Crop from 1024X1024_40X_homogenized/');
rgStackDir = dir(strcat(rgStackPath, '*.tif'));

for index = 1:size(rgStackDir, 1)
%     
%     fileName = rgStackDir(index).name;
%     disp(fileName);
%     
%     refImg = readStackTif(strcat(referencePath, fileName));
%     toHomoImg = readStackTif(strcat(toHomogenizePath, fileName));
%     
%     se = strel('sphere', 5);
%     toHomoImg = imdilate(toHomoImg, se);
%   
%     homoRedImg = imresize3(toHomoImg, [size(refImg, 1), size(refImg,2), size(refImg, 3)], 'nearest');
%     
%     homoRedImg = homoRedImg.*(refImg/255);
%     
%     writeStackTif(homoRedImg, strcat(savePath, fileName));

    
    
    fileName = rgStackDir(index).name;
    fileName = strsplit(fileName, '.tif');
    fileName = fileName{1};
    disp(fileName);
    
    [originalImage, imgInfo] = readStackTif(strcat(rgStackPath, fileName, '.tif'));
    [originalImage2, ~] = readStackTif(strcat(rgStackPath2, fileName, '.tif'));

    %% Extract pixel-micron relation
    xResolution = imgInfo(1).XResolution;
    yResolution = imgInfo(1).YResolution;
    spacingInfo = strsplit(imgInfo(1).ImageDescription, 'spacing=');
    spacingInfo = strsplit(spacingInfo{1}, '\n');
    spacingInfo = spacingInfo{1};
    spacingInfo = strsplit(spacingInfo, 'ImageJ=');
    spacingInfo = strsplit(spacingInfo{2}, 't');
    z_pixel = str2num(spacingInfo{1});
    x_pixel = 1/xResolution;
    y_pixel = 1/yResolution;

    %% Get original image size
    shape = size(originalImage);

    %% Make homogeneous
    numRows = shape(1);
    numCols = shape(2);
    numSlices = round(shape(3)*(z_pixel/x_pixel));

    originalImage2 = imresize3(originalImage2, [numRows, numCols, numSlices]);
    originalImage2 = imresize3(originalImage2, 1);
    homogenized = originalImage2;
    writeStackTif(homogenized, strcat(savePath, fileName, '.tif'));

end
