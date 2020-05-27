function [ P ] = RecogniseFace(input_img, featureType, classifierName) 

%clear all;clc;close all;
 img = imread(input_img);
%% Preprocessing
myFaceDetector = vision.CascadeObjectDetector();
 myFaceDetector.MinSize = [80 80]; 
 myFaceDetector.MaxSize = [400 400];
BBOX = step(myFaceDetector, img); 
[m,n]=size(BBOX);
P=zeros(m,3);
B = insertObjectAnnotation(img,'rectangle',BBOX,'Face');
 imshow(B)
%% HOG MLP
if strcmpi(featureType, 'HOG') && strcmpi(classifierName, 'MLP')
   
load ('MLP_HOG.mat');

for x = 1:m
    face2 = img(BBOX(x,2):BBOX(x,2)+BBOX(x,4),BBOX(x,1):BBOX(x,1)+BBOX(x,3),: );%,BBOX(x,2),BBOX(x,3),BBOX(x,4));
    face = imresize(face2, [70 70]);
    %face = rgb2gray(face);
    
    queryFeatures = extractHOGFeatures(face);
    
    outPuts = net(queryFeatures');
    [value label] = max(outPuts(:,1));
    actualLabel = M(label)
    figure('Name',num2str(actualLabel));
    imshow(face);
     mid_point = BBOX(x,4)/2;
     mid_x = BBOX(x,1) + mid_point;
     mid_y = BBOX(x,2) + mid_point;
     P(x, :) = [actualLabel mid_x mid_y];
end
Fig1=  insertObjectAnnotation(img,'rectangle',BBOX,P(:,1),'FontSize',42);
  imshow(Fig1);

%% HOG SVM
elseif strcmpi(featureType, 'HOG') && strcmpi(classifierName, 'SVM')
    % load compacted classifier
    load('SVM_HOG.mat');
    % step through each face, and populate matrix P
    for x = 1:m
        face = img(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [70 70]);
        face = rgb2gray(face);
        queryFeatures = extractHOGFeatures(face);
        label = predict(faceClassifier, queryFeatures);
        label = str2num(label{1})
        %figure('Name',num2str(label));imshow(face);
       % emotion = predict(emotionClassifier, queryFeatures);
       % emotion = str2num(emotion{1});
        
         mid_point = BBOX(x,4)/2;
         mid_x = BBOX(x,1) + mid_point;
         mid_y = BBOX(x,2) + mid_point;
         P(x, :) = [label mid_x mid_y];
    end
    
  Fig1=  insertObjectAnnotation(img,'rectangle',BBOX,P(:,1),'FontSize',42);
  imshow(Fig1);

    
%% SURF MLP
elseif strcmpi(featureType, 'SURF') && strcmpi(classifierName, 'MLP')
load('MLP_SURF.mat');
    % step through each face, and populate matrix P
    for x = 1:m
        face = img(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [80 80]);
        figure;imshow(face);
        queryFeatures = encode(bag,face).';
        outPuts = net(queryFeatures);
        [value label] = max(outPuts(:,1));
        Label=M(label);
%         queryFeatures = extractHOGFeatures(face);
%         emotion = predict(emotionClassifier, queryFeatures);
%         emotion = str2num(emotion{1});
%         
         mid_point = BBOX(x,4)/2;
         mid_x = BBOX(x,1) + mid_point;
         mid_y = BBOX(x,2) + mid_point;
         P(x, :) = [Label, mid_x, mid_y]; %center point, %center point]
    end
    Fig1=  insertObjectAnnotation(img,'rectangle',BBOX,P(:,1),'FontSize',42);
  imshow(Fig1);
%% SURF SVM
elseif strcmpi(featureType, 'SURF') && strcmpi(classifierName, 'SVM')
load('SVM_SURF.mat');
    % step through each face, and populate matrix P
    for x = 1:m
        face = img(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        figure;imshow(face);
        face = imresize(face, [70 70]);
        label = predict(categoryClassifier, face);
        
%         queryFeatures = extractHOGFeatures(face);
%         emotion = predict(emotionClassifier, queryFeatures);
%         emotion = str2num(emotion{1});
%         
         mid_point = BBOX(x,4)/2;
         mid_x = BBOX(x,1) + mid_point;
         mid_y = BBOX(x,2) + mid_point;
         P(x,1) = str2double(categoryClassifier.Labels{label});
        P(x, 2:3) = [ mid_x, mid_y]; %center point, %center point]
    end

%% 
Fig1=  insertObjectAnnotation(img,'rectangle',BBOX,P(:,1),'FontSize',42);
     imshow(Fig1);
elseif strcmpi(featureType, 'NIL') && strcmpi(classifierName, 'CNN')
    load('AlexNet.mat');
    %load('HOG_SVM.mat','alexnet_labels');
    for x = 1:m
        face = img(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
%         figure;imshow(face);
        face = imresize(face, [227 227]);
        [label,~] = classify(alexNet_mdl, face)
         %Label=str2double(label)
         mid_point = BBOX(x,4)/2;
         mid_x = BBOX(x,1) + mid_point;
         mid_y = BBOX(x,2) + mid_point;
        P(x, 1) = label;
        P(x,2:3) = [mid_x mid_y]
    end
     Fig1=  insertObjectAnnotation(img,'rectangle',BBOX,P(:,1),'FontSize',42);
     imshow(Fig1);
end
