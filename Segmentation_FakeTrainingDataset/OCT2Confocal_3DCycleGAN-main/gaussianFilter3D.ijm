// Set input and output directories
dirA = getDirectory("Choose Source Directory");
dirB = getDirectory("Choose Destination Directory");

// Set Gaussian filter parameters (sigmaX, sigmaY, sigmaZ)
sigmaX = 1.0;
sigmaY = 1.0;
sigmaZ = 1.0;

// Get list of files
list = getFileList(dirA);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".tiff") || endsWith(list[i], ".jpg") || endsWith(list[i], ".png")) {
        open(dirA + list[i]);
        run("Gaussian Blur 3D...", "x=" + sigmaX + " y=" + sigmaY + " z=" + sigmaZ);
        saveAs("Tiff", dirB + list[i]);
        close();
    }
}

