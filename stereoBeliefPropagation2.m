% Stereo Matching using Belief Propagation (with Synchronous message update schedule)
% Computes a disparity map from a rectified stereo pair using Belief Propagation

% Parameters
dispLevels = 16; %disparity range: 0 to dispLevels-1
iterations = 60;
lambda = 5; %weight of smoothness cost
trunc = 2; %truncation of smoothness cost

% Load left and right images in grayscale
leftImg = rgb2gray(imread('left.png'));
rightImg = rgb2gray(imread('right.png'));

% Apply a Gaussian filter
leftImg = imgaussfilt(leftImg,0.6,'FilterSize',5);
rightImg = imgaussfilt(rightImg,0.6,'FilterSize',5);

% Get the size
[rows,cols] = size(leftImg);

% Compute pixel-based matching cost (data cost)
rightImgShifted = zeros(rows,cols,dispLevels);
for d = 0:dispLevels-1
    rightImgShifted(:,d+1:end,d+1) = rightImg(:,1:end-d);
end
dataCost = abs(double(leftImg)-rightImgShifted);

% Compute smoothness cost
d = 0:dispLevels-1;
smoothnessCost = lambda*min(abs(d-d.'),trunc);
smoothnessCost4d = zeros(1,1,dispLevels,dispLevels);
smoothnessCost4d(1,1,:,:) = smoothnessCost;

% Initialize messages
msgFromUp = zeros(rows,cols,dispLevels);
msgFromDown = zeros(rows,cols,dispLevels);
msgFromRight = zeros(rows,cols,dispLevels);
msgFromLeft = zeros(rows,cols,dispLevels);

figure
energy = zeros(iterations,1);

% Start iterations
for it = 1:iterations

    % Create messages to up
    msgToUp = dataCost + msgFromDown + msgFromRight + msgFromLeft;
    msgToUp = permute(min(msgToUp+smoothnessCost4d,[],3),[1,2,4,3]);
    msgToUp = msgToUp-min(msgToUp,[],3); % normalize
    
    % Create messages to down
    msgToDown = dataCost + msgFromUp + msgFromRight + msgFromLeft;
    msgToDown = permute(min(msgToDown+smoothnessCost4d,[],3),[1,2,4,3]);
    msgToDown = msgToDown-min(msgToDown,[],3); % normalize
    
    % Create messages to right
    msgToRight = dataCost + msgFromUp + msgFromDown + msgFromLeft;
    msgToRight = permute(min(msgToRight+smoothnessCost4d,[],3),[1,2,4,3]);
    msgToRight = msgToRight-min(msgToRight,[],3); % normalize
    
    % Create messages to left
    msgToLeft = dataCost + msgFromUp + msgFromDown + msgFromRight;
    msgToLeft = permute(min(msgToLeft+smoothnessCost4d,[],3),[1,2,4,3]);
    msgToLeft = msgToLeft-min(msgToLeft,[],3); % normalize

    % Send messages
    msgFromDown = [msgToUp(2:end,:,:);zeros(1,cols,dispLevels)]; %shift up
    msgFromUp = [zeros(1,cols,dispLevels);msgToDown(1:end-1,:,:)]; %shift down
    msgFromLeft = [zeros(rows,1,dispLevels),msgToRight(:,1:end-1,:)]; %shift right
    msgFromRight = [msgToLeft(:,2:end,:),zeros(rows,1,dispLevels)]; %shift left

    % Compute belief
    %belief = dataCost + msgFromUp + msgFromDown + msgFromRight + msgFromLeft; % Standard belief computation
    belief = msgFromUp + msgFromDown + msgFromRight + msgFromLeft; % Without dataCost (larger energy but better results)
    
    % Compute the disparity map
    [~,ind] = min(belief,[],3);
    dispMap = ind-1;
    
    % Compute energy
    [row,col] = ndgrid(1:size(ind,1),1:size(ind,2));
    linInd = sub2ind(size(dataCost),row,col,ind);
    dataEnergy = sum(sum(dataCost(linInd)));
    row = [reshape(ind(:,1:end-1),[],1);reshape(ind(1:end-1,:),[],1)];
    col = [reshape(ind(:,2:end),[],1);reshape(ind(2:end,:),[],1)];
    linInd = sub2ind(size(smoothnessCost),row,col);
    smoothnessEnergy = sum(smoothnessCost(linInd));
    energy(it) = dataEnergy+smoothnessEnergy;
    
    % Normalize the disparity map for display
    scaleFactor = 256/dispLevels;
    dispImg = uint8(dispMap*scaleFactor);

    % Show disparity map
    imshow(dispImg)
    
    % Show energy and iteration
    fprintf('iteration: %d/%d, energy: %d\n',it,iterations,energy(it))
end

% Show convergence graph
figure
plot(1:iterations,energy,'bo-')
xlabel('Iterations')
ylabel('Energy')

% Save disparity map
imwrite(dispImg,'disparity.png')