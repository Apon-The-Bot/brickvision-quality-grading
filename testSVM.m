clc; clear; close all;

% Load trained model and normalization parameters
load('BrickSVMModel.mat');  % contains model, mu, sigma

% Test image
testImagePath = 'D:\4.jpeg'; % Change path accordingly
testImg = imresize(imread(testImagePath), [256, 256]);
grayImg = rgb2gray(testImg);
R = double(testImg(:,:,1)); G = double(testImg(:,:,2)); B = double(testImg(:,:,3));
hsvImg = rgb2hsv(testImg);
ycbcrImg = rgb2ycbcr(testImg);

% Feature extraction (same as training)
colorFeatures = [mean(R(:)), mean(G(:)), mean(B(:)), ...
                 mean(hsvImg(:,:,1), 'all'), mean(hsvImg(:,:,2), 'all'), mean(hsvImg(:,:,3), 'all'), ...
                 mean(ycbcrImg(:,:,1), 'all'), mean(ycbcrImg(:,:,2), 'all'), mean(ycbcrImg(:,:,3), 'all')];

glcm = graycomatrix(grayImg, 'Offset', [1 0; 0 1], 'Symmetric', true);
stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
textureFeatures = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];

lbpFeatures = extractLBPFeatures(grayImg, 'Upright', false);
bw = bwareafilt(imbinarize(grayImg), 1);
region = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'Extent');
if ~isempty(region)
    shapeFeatures = [region.Area, region.Perimeter, region.Eccentricity, region.Solidity, region.Extent];
else
    shapeFeatures = zeros(1, 5);
end

% Custom Features
hueChannel = hsvImg(:,:,1);
pureRedHue = 0;
hueDistance = abs(hueChannel - pureRedHue);
hueDistance = min(hueDistance, 1 - hueDistance);
MHD_Red = mean(hueDistance(:));
CTF = 0.299 * mean(R(:)) + 0.587 * mean(G(:)) + 0.114 * mean(B(:));
[Gx, Gy] = imgradientxy(grayImg);
[Gmag, ~] = imgradient(Gx, Gy);
CGM = mean(Gmag(:));
customFeatures = [MHD_Red, CTF, CGM];

% Combine and normalize
testFeatures = [colorFeatures, shapeFeatures, textureFeatures, lbpFeatures, customFeatures];
testFeatures = (testFeatures - mu) ./ sigma;

% Predict
predictedLabel = predict(model, testFeatures);

% Show result
imshow(testImg);
title(['Predicted Brick Grade: ', char(predictedLabel)]);
disp(['Predicted Brick Grade: ', char(predictedLabel)]);
