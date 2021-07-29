clear all;
[ vfilename, vpathname ] = uigetfile( 'dataset\*.avi', 'Select an video' );
I=VideoReader(strcat( vpathname, vfilename ));
nFrames = I.numberofFrames;
vidHeight =  I.Height;
vidWidth =  I.Width;
mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);
for k = 1: nFrames
    mov(k).cdata = read( I, k);
   mov(k).cdata = imresize(mov(k).cdata,[256,256]);
    imwrite(mov(k).cdata,['Frames\',num2str(k),'.jpg']);
end
implay([vpathname vfilename]);
mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);
for k = 1: nFrames
    mov(k).cdata = read( I, k);
    imwrite(mov(k).cdata,['Frames\',num2str(k),'.jpg']);
end
for i = 1: 40
    im=imread(['Frames\',num2str(i),'.jpg']); 
    figure(1)
    subplot(5,8,i),imshow(im);
    axis off;
end
title(' Frame conversion');

for i = 1: nFrames
    im=imread(['Frames\',num2str(i),'.jpg']); 
end
figure(2),imshow(im);title('original image');
J=rgb2gray(im);figure(3),imshow(J);title('gray image');
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 40);
videoreader= vision.VideoFileReader(([vpathname vfilename]));
for i = 1:40
    frame = step(videoreader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end
figure(4); imshow(frame); title('Video Frame');
figure(5); imshow(foreground); title('Foreground');
A =im2bw(im);figure(6),imshow(A);title('binary image');
B=medfilt2(J,[5 5]);figure(7),imshow(B);title('median filter image');
for i = 1:nFrames
im = imread(['Frames\',num2str(i),'.jpg']);
im_gray=rgb2gray(im);
imedge=edge(im_gray,'canny',[0.1 0.2]);
figure(8),imshow(imedge);
end 
feature=mean(B);
Testfeature=feature(1,1:100);
load Trainfeature
load label
inputs =Trainfeature ;
targets =Trainfeature ;
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
% View the Network
view(net)
%Neuron weights
w = [4 -2];
%Neuron bias
b=-3;
func='tansig';
p=[2 3];
activation_potential=p*w'+b;
[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b );
z = reshape(z,length(p1),length(p2));
figure(9),plot3(p1,p2,z)
grid on
xlabel('Input 1')
ylabel('Input 2')
zlabel('Neuron output')
net = network(   ...
     1,          ...
     2,          ...
    [1; 0],      ...
    [1; 0],      ...
    [0 0; 1 0],  ...
    [0 1]        ...
    );
    view(net);
T = ind2vec(label);
net = newpnn(Trainfeature',T);
Y = net(Testfeature');
class = vec2ind(Y);
figure('Name','Recognized Action','NumberTitle','off');
imshow('Frames\8.jpg')
hold on;
if (class==1)
   text(8, 18, 'Boxing', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==2)
   text(8, 18, 'Jogging', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==3)
   text(8, 18, 'Hand Waving', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==4)
   text(8, 18, 'Hand Clapping', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==5)
   text(8, 18, 'Jumping', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==6)
   text(8, 18, 'Running', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==7)
   text(8, 18, 'Cycling', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
else if(class==8)
   text(8, 18, 'Surfing', 'Color','r','fontname','Times New Roman', 'FontWeight','bold', 'FontSize',20);
    end
    end
    end
    end
    end
    end
    end
    end
