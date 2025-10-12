function [s] = getSynthethicCyst(cystRadius, lumenRadius, nCells)
    
    lumenRadius = 120;
    cystRadius = 200;
    nCells = 100;
    
    s = randn(3,nCells);
    r = (rand(1,nCells)*(cystRadius^2-lumenRadius^2)+lumenRadius^2).^(1/2);
    c = r./sqrt(sum(s.^2,1));
    s = bsxfun(@times, s, c);
    s = round((s+cystRadius));
%     space = zeros([cystRadius*2*100, cystRadius*2*100, cystRadius*2*100]);
    space = zeros([512, 512, 512]);
    center = zeros([3,nCells])+256-cystRadius;

    s = s+center;
    
    for ix = 1:nCells
        space(s(1, ix), s(2, ix), s(3, ix)) = 1;
    end
    
    space = imresize3(space, [512, 512, 512]);
    se = strel('sphere', 5);
    space = imdilate(space, se);
    
    labelvolshow(bwlabeln(space));
    
    strelCyst = strel("sphere", cystRadius);
    strelLumen = strel("sphere", lumenRadius);
    
    %%
    
    voronoiSpaceCyst = zeros([512, 512, 512]);
    [meshX,meshY,meshZ] = meshgrid(1:512, 1:512, 1:512);
    xc = 256; yc = 256; zc = 256; % the center of sphere
    cystSphere = (meshX-xc).^2 + (meshY-yc).^2 + (meshZ-zc).^2 <=(cystRadius*cystRadius);
    voronoiSpaceCyst(cystSphere) = 1; % set to zero
    
    voronoiSpaceLumen = zeros([512, 512, 512]);
    lumenSphere = (meshX-xc).^2 + (meshY-yc).^2 + (meshZ-zc).^2 <=(lumenRadius*lumenRadius);
    voronoiSpaceLumen(lumenSphere) = 1; % set to zero
    
    cellSpace = voronoiSpaceCyst-voronoiSpaceLumen;
    seeds = bwlabeln(space);
    
    cellSpace = imresize3(cellSpace, 0.3, 'nearest');
    seeds = imresize3(seeds, 0.3, 'nearest');
        
    %Voronoi a partir de máscara y semillas
    voronoiCyst = VoronoizateCells(cellSpace,seeds);
    
    
    %resize y guardar cosas
    voronoiCyst = imresize3(voronoiCyst, [255, 255, 255], 'nearest');
    
    writeStackTif(voronoiCyst, '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/voronoiModelCyst.tif')
    
    outlines = getCellOutlines(voronoiCyst);
    
    writeStackTif(outlines, '/media/pedro/6TB/jesus/voronoiBasedGenerativeModel/voronoiModelCystOutlines.tif')

    
    volumeSegmenter(voronoiCyst, voronoiCyst);
    %%
    
    voronoiSpace = imdilate(voronoiSpace, strelCyst);
    voronoiSpace_aux = imdilate(voronoiSpace_aux, strelLumen);
    
    cellSpace = voronoiSpace-voronoiSpace_aux;

    


    
end
    


