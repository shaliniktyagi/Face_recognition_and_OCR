path = fullfile('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\newset');
files = dir(path)
Dirs = [files.isdir]
subdirectories = files(Dirs)

%%
tic
for i = 1:length(subdirectories)
        % Get subfolders under originalimagefolder
    subfolder=fullfile(path,subdirectories(i).name);%i=3
    imagefiles=dir(fullfile(subfolder,'\*.jpg'));
    %change_dir = strcat('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\individual_pic_folder\',num2str(i));
    %cd('change_dir');
    %delete *crop*;
    %cd ..;
    for j = 1:length(imagefiles)
        F = fullfile(subfolder,imagefiles(j).name);%j=1
        I = imread(F);
        disp(strcat('crop face from image: ',subdirectories(i).name,'\',imagefiles(j).name));
        %imshow(I);
        myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
        %myFaceDetector.MergeThreshold = 7;
        %BBOX = myFaceDetector(I);
        BBOX = step(myFaceDetector, I);
        if isempty(BBOX) == 0
        B = insertObjectAnnotation(I,'rectangle',BBOX,'Face');
        %imshow(B);
        cropfacesavepath=strcat(path,'\',subdirectories(i).name,'\','image_',num2str(j),'_crop_',num2str(j),'.jpg');%i=3, j=1
        cropimage = imcrop(I, BBOX(1, :));
        P = imresize(cropimage, [227, 227]);
        %P = imresize(cropimage, [70, 70]);
        %J = rgb2gray(P);
             
        %imshow(J);
        imwrite(P,cropfacesavepath);
        %disp(cropfacesavepath); 
        end
    end
end
toc        

        
        
