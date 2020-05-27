path = fullfile('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\individual_pic_folder');
files = dir(path)
Dirs = [files.isdir]
subdirectories = files(Dirs)
%%

for i = 1 : length(subdirectories)

        % Get subfolders under originalimagefolder

        subfolder=fullfile(path,subdirectories(i).name);

        

        % Get list of mp4 video files in folder

        videofiles=dir(fullfile(subfolder,'\*.mp4'));

        

        % Loop through the videofiles one-by-one

        

        % define number of frames to extract --> given number or videos

        % available we need more frames where fewer videos are available

        

        

        

        

        for v = 1 : length(videofiles)

            

            % Get the full video file path in subfolder

            videofilename = fullfile(subfolder,videofiles(v).name);

            

            disp(strcat('Extracting images from video: ',subdirectories(i).name,'\',videofiles(v).name));

            

            % This code is adapted from here - https://uk.mathworks.com/matlabcentral/answers/48797-how-to-extract-frames-of-a-video

            Vid=VideoReader(videofilename); % Read Video File - mp4 format

        

            for img = 1:10; % Read every nth frame where n=frame_step

    

                videofilesavepath=strcat(path,'\',subdirectories(i).name,'\','video_',num2str(v),'_frame_',num2str(img),'.jpg');

    

                videoimage = read(Vid, img);

    

                %imshow(videoimage);

    

                imwrite(videoimage,videofilesavepath);

        

                %movie(img)

            end

            

        end

        

    end

    

