inputDir = "/home/jesus/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main/dataset/dataset_20250522/testA/";
outputDir = "/home/jesus/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main/dataset/dataset_20250522/testA_y/";

list = getFileList(inputDir);
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {
        open(inputDir + list[i]);
        title = getTitle();

        // Resize using Size (fast, but shows dialog unless in batch mode)
        run("Size...", "width=80 height=80 depth=64 constrain average interpolation=Bilinear");

        // Convert to 8-bit
        run("8-bit");

        // Binarize: set all pixels > 1 to 255
        setThreshold(2, 255);
        run("Make Binary");

        // Save as TIF
        saveAs("Tiff", outputDir + title);
        close();
    }
}
