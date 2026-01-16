% Stereo Matching using Belief Propagation (Synchronous) - a different aproach
% ------------------------------------------------------------
dispLevels = 16; %disparity range: 0 to dispLevels-1
iterations = 60;
lambda = 5; %weight of smoothness cost
%smoothness cost computation: min(abs(d1-d2),2)*lambda

% Load left and right images in grayscale
leftImg = rgb2gray(imread('left.png'));
rightImg = rgb2gray(imread('right.png'));

% Apply a Gaussian filter
leftImg = imgaussfilt(leftImg,0.6,'FilterSize',5);
rightImg = imgaussfilt(rightImg,0.6,'FilterSize',5);

% Convert to double
leftImg = double(leftImg);
rightImg = double(rightImg);

% Get the size
[rows,cols] = size(leftImg);

% Compute pixel-based matching cost (data cost)
rightImgShifted = zeros(rows,cols,dispLevels);
for d = 0:dispLevels-1
	rightImgShifted(:,d+1:end,d+1) = rightImg(:,1:end-d);
end
dataCost = abs(leftImg-rightImgShifted);

% Initialize messages
msgFromUp = zeros(rows,cols,dispLevels);
msgFromDown = zeros(rows,cols,dispLevels);
msgFromRight = zeros(rows,cols,dispLevels);
msgFromLeft = zeros(rows,cols,dispLevels);

msgToUp = Inf(rows,cols,dispLevels+2);
msgToDown = Inf(rows,cols,dispLevels+2);
msgToRight = Inf(rows,cols,dispLevels+2);
msgToLeft = Inf(rows,cols,dispLevels+2);

costs = zeros(rows,cols,3);
energy = zeros(iterations,1);
figure

% Start iterations
for it = 1:iterations

    % Compute messages (step 1)
    msgToUp(:,:,2:end-1) = dataCost + msgFromDown + msgFromRight + msgFromLeft;
    msgToDown(:,:,2:end-1) = dataCost + msgFromUp + msgFromRight + msgFromLeft;
    msgToRight(:,:,2:end-1) = dataCost + msgFromUp + msgFromDown + msgFromLeft;
    msgToLeft(:,:,2:end-1) = dataCost + msgFromUp + msgFromDown + msgFromRight;

    % Find minimum costs
    minMsgToUp = min(msgToUp,[],3);
    minMsgToDown = min(msgToDown,[],3);
    minMsgToRight = min(msgToRight,[],3);
    minMsgToLeft = min(msgToLeft,[],3);

    % Compute messages (step 2)
    for i = 1:dispLevels
        % Messages to up
        costs(:,:,1) = msgToUp(:,:,i+1);
        costs(:,:,2) = min(msgToUp(:,:,i),msgToUp(:,:,i+2))+lambda;
        costs(:,:,3) = minMsgToUp+2*lambda;
        msgFromDown(:,:,i) = min(costs,[],3)-minMsgToUp;
        
        % Messages to down
        costs(:,:,1) = msgToDown(:,:,i+1);
        costs(:,:,2) = min(msgToDown(:,:,i),msgToDown(:,:,i+2))+lambda;
        costs(:,:,3) = minMsgToDown+2*lambda;
        msgFromUp(:,:,i) = min(costs,[],3)-minMsgToDown;
        
        % Messages to right
        costs(:,:,1) = msgToRight(:,:,i+1);
        costs(:,:,2) = min(msgToRight(:,:,i),msgToRight(:,:,i+2))+lambda;
        costs(:,:,3) = minMsgToRight+2*lambda;
        msgFromLeft(:,:,i) = min(costs,[],3)-minMsgToRight;
        
        % Messages to left
        costs(:,:,1) = msgToLeft(:,:,i+1);
        costs(:,:,2) = min(msgToLeft(:,:,i),msgToLeft(:,:,i+2))+lambda;
        costs(:,:,3) = minMsgToLeft+2*lambda;
        msgFromRight(:,:,i) = min(costs,[],3)-minMsgToLeft;
    end

    % Fix messages
    msgFromDown = [msgFromDown(2:end,:,:);zeros(1,cols,dispLevels)]; %shift up
    msgFromUp = [zeros(1,cols,dispLevels);msgFromUp(1:end-1,:,:)]; %shift down
    msgFromLeft = [zeros(rows,1,dispLevels),msgFromLeft(:,1:end-1,:)]; %shift right
    msgFromRight = [msgFromRight(:,2:end,:),zeros(rows,1,dispLevels)]; %shift left

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
    smoothnessEnergyHorizontal = sum(sum(min(abs(ind(:,1:end-1)-ind(:,2:end)),2)*lambda));
    smoothnessEnergyVertical = sum(sum(min(abs(ind(1:end-1,:)-ind(2:end,:)),2)*lambda));
    energy(it) = dataEnergy+smoothnessEnergyHorizontal+smoothnessEnergyVertical;

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