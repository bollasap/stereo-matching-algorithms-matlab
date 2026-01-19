% Stereo Matching using Block Matching
% Computes a disparity map from a rectified stereo pair using Block Matching

% Parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
windowSize = 5;

% Load left and right images in grayscale
leftImg = rgb2gray(imread('left.png'));
rightImg = rgb2gray(imread('right.png'));

% Apply a Gaussian filter
leftImg = imgaussfilt(leftImg,0.6,'FilterSize',5);
rightImg = imgaussfilt(rightImg,0.6,'FilterSize',5);

% Get the size
[rows,cols] = size(leftImg);

% Compute pixel-based matching cost
rightImgShifted = zeros(rows,cols,dispLevels);
for d = 0:dispLevels-1
    rightImgShifted(:,d+1:end,d+1) = rightImg(:,1:end-d);
end
dataCost = abs(double(leftImg)-rightImgShifted);

% Aggregate the matching cost
dataCost = imboxfilt3(dataCost,[windowSize windowSize 1]);

% Compute the disparity map
[~,ind] = min(dataCost,[],3);
dispMap = ind-1;

% Normalize the disparity map for display
scaleFactor = 256/dispLevels;
dispImg = uint8(dispMap*scaleFactor);

% Show disparity map
figure; imshow(dispImg)

% Save disparity map
imwrite(dispImg,'disparity.png')