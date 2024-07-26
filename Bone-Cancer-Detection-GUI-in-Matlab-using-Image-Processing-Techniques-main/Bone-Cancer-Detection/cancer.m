function varargout = cancer(varargin)
% CANCER M-file for cancer.fig
%      CANCER, by itself, creates a new CANCER or raises the existing
%      singleton*.
%
%      H = CANCER returns the handle to a new CANCER or the handle to
%      the existing singleton*.
%
%      CANCER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CANCER.M with the given input arguments.
%
%      CANCER('Property','Value',...) creates a new CANCER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before cancer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to cancer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help cancer

% Last Modified by GUIDE v2.5 13-Apr-2021 09:01:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cancer_OpeningFcn, ...
                   'gui_OutputFcn',  @cancer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before cancer is made visible.
function cancer_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to cancer (see VARARGIN)

% Choose default command line output for cancer
handles.output = hObject;
ah = axes('unit', 'normalized', 'position', [0 0 1 1]); 
% import the background image and show it on the axes
bg = imread('bg.jpg'); imagesc(bg);
% prevent load over the background and turn the axis off
set(ah,'handlevisibility','off','visible','off')
% making sure the background is behind all the other uicontrols
uistack(ah, 'bottom');
global input_button_status
input_button_status=0;
global fcmeans_check_status
cla(handles.axes1);
cla(handles.axes10);
cla(handles.axes11);
cla(handles.axes12);

cla(handles.axes15);
cla(handles.axes16);
cla(handles.axes17);
format LongG;
set(handles. pushbutton2,'Enable','off');
set(handles. pushbutton4,'Enable','off');
set(handles. checkbox1,'Enable','off');
set(handles. checkbox2,'Enable','off');

if evalin( 'base', 'exist(''kmeans_input_fig'',''var'') == 1' )
close('Name','K Means Clustering Inputs')
clear kmeans_input_fig
end
if evalin( 'base', 'exist(''kmeans_result_fig'',''var'') == 1' )
    close('Name','K Means Clustering Results')
    clear kmeans_result_fig
end
if evalin( 'base', 'exist(''fcmeans_input_fig'',''var'') == 1' )
close('Name','Fuzzy C Means Clustering Results')
clear fcmeans_input_fig
end
clear img
clc

% Update handles structure
guidata(hObject, handles);

% set(handles.axes19)
% logo=imread('Logo.jpg');
% imshow(logo);
% Update handles structure
guidata(hObject, handles);


% UIWAIT makes cancer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = cancer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global img
global input_button_status
clear segmentatation_status


clc
code_path=cd;
dataset_path='DS';
cd(dataset_path)

[filename, pathname] = uigetfile('*.jpg', 'Pick an Image');

cd(code_path)
if isequal(filename,0) | isequal(pathname,0)
   warndlg('File is not selected');
else
global bpath, global bname, global bext
    [bpath,bname,bext]=fileparts([pathname,filename]);

    set(handles. checkbox1,'Enable','on');
    set(handles. checkbox2,'Enable','on');
    set(handles. pushbutton2,'Enable','on');
      input_button_status=1;
      img=imread([pathname,filename]);
      [bpath,bname,bext]=fileparts([pathname,filename]);
      axes(handles.axes1)
      
%     set( handles.edit2,'BackgroundColor','blue')
%     set(handles.edit1,'String',newValue);
      imshow(img);  
      title(bname);
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global input_button_status
%% Checking if Input File is provided or not ??
if isequal(input_button_status,1)
global img
global segmented_images_kmeans
global kmeans_check_status
global fuzzycmeans_check_status
global segmentatation_status
%% Check if K Means Check Box is Selected
if handles.checkbox1.Value
% % % % Color Space Conversion
%   Conversion Using MakeCForm
% makecform('srgb2lab') uses WhitePoint 'ICC' by default.     
    cform = makecform('srgb2lab');
    lab_he = applycform(img,cform);
%     Conversion Using RGB2LAB Function
%     rgb2lab() uses whitepoint 'D65' by default.
    lab_he1=rgb2lab(img);
% kmeans_input_fig=figure('Name','K Means Clustering Inputs');

figure(1)
subplot(131),imshow(img),title('Input Image');
subplot(132),imshow(lab_he),title('Make C Form Converted Image');
subplot(133),imshow(lab_he1),title('Lab Image Using Bultin Function');

%%  Step 3: Classify the Colors in 'a*b*' Space Using K-Means Clustering
ab = double(img(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);


nColors = 3;

 opts = statset('Display','iter');
%% New KMeans
[cluster_idx1_kmeans, Clusters1_kmeans,SUMD1, D1]= fun_KMeans(ab,nColors, 'Options',opts,...
                    'Replicates',3);

%% Improved K means
[kmeans_cluster_idx,kmeans_clusters2,kmeans_EachCentroid]=fun_kmeans3(ab,3);

disp(['K Means 1st Cluster Centroid : ',num2str(kmeans_clusters2(1,1))]);
disp(['K Means 2nd Cluster Centroid : ',num2str(kmeans_clusters2(2,1))]);
disp(['K Means 3rd Cluster Centroid : ',num2str(kmeans_clusters2(3,1))]);

Final_Centroid={kmeans_clusters2(1,1),kmeans_clusters2(2,1),kmeans_clusters2(3,1)};

Results_Kmeans_Centroid_final=table();
Results_Kmeans_Centroid_final.Cluster1=kmeans_clusters2(1,1);
Results_Kmeans_Centroid_final.Cluster2=kmeans_clusters2(2,1);
Results_Kmeans_Centroid_final.Cluster3=kmeans_clusters2(3,1);

Results_kmeans_Each_Centroid=table();
Results_kmeans_Each_Centroid.Cluster1=kmeans_EachCentroid(:,1);
Results_kmeans_Each_Centroid.Cluster2=kmeans_EachCentroid(:,2);
Results_kmeans_Each_Centroid.Cluster3=kmeans_EachCentroid(:,3);

figure(666),
    plot(ab(kmeans_cluster_idx==1,1),ab(kmeans_cluster_idx==1,2),'r.', ...
        ab(kmeans_cluster_idx==2,1),ab(kmeans_cluster_idx==2,2),'g.',...
        ab(kmeans_cluster_idx==3,1),ab(kmeans_cluster_idx==3,2),'b.',...
        kmeans_clusters2(:,1),kmeans_clusters2(:,2),'kx'),
    legend({'Cluster 1','Cluster 2','Cluster 3'});
        title('Plot of Improved K Means');

pixel_labels = reshape(kmeans_cluster_idx,nrows,ncols);
axes(handles.axes10)
imshow(pixel_labels,[])
    
% figure(55),
% subplot(121),imshow(pixel_labels,[]),title('Stock');
% subplot(122),imshow(pixel_labels_2,[]),title('New');

%       title 'Input Image'     

% set(handles.pushbutton3,'Rotate',90)


%%  Step 5: Create Images that Segment Image by Color.
segmented_images_kmeans = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);
for k = 1:nColors
    color = img;
    color(rgb_label ~= k) = 0;
    segmented_images_kmeans{k} = color;
end

global segmented_images_kmeans
global selected_cluster_kmeans

cluster1=im2bw(segmented_images_kmeans{1});
cluster2=im2bw(segmented_images_kmeans{2});
cluster3=im2bw(segmented_images_kmeans{3});

figure(2),
subplot(231),imshow(double(cluster1)),title('KMeans Clutser 1');
subplot(232),imshow(double(cluster2)),title('KMeans Clutser 2');
subplot(233),imshow(double(cluster3)),title('KMeans Clutser 3');

%% Selecting Best Cluster (User Defined Function)
selected_cluster_kmeans=fun_find_best_cluster(cluster1,cluster2,cluster3,1);

figure(2)
subplot(235),imshow(double(selected_cluster_kmeans)),title('K Means Selected Cluster');
axes(handles.axes11)
imshow(double(selected_cluster_kmeans))
segmentatation_status=1;
kmeans_check_status=1;
set(handles. pushbutton4,'Enable','on');
% disp('Running K Means Algorithm');
else
% disp('K Means Not Selected');
kmeans_check_status=0;
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%% % % %  Check if Fuzzy C Means Check Box is selected or not ????

if handles.checkbox2.Value
    fuzzycmeans_check_status=1;
    fcmeans_check_status=1;
%%  Fuzzy C Means Function
% % % % % % % % % % % % % % % % % 
global img
I=double(min(img(:)):max(img(:)));
img_1=img(:,:,1);
%% Fuzzy C Means Based Segmentation

[Fuzzy_Centroid_Final,fuzzy_U,fuzzy_LUT,fuzzy_H,fuzzy_centroid_details,fuzzy_Distance_Each]=fun_FastFCMeans(img_1,3); % perform segmentation
%% Visualize the segmentation
Lut_based_image=fun_LUT2label(img_1,fuzzy_LUT);

Lrgb=zeros([numel(Lut_based_image) 3],'uint8');
for i=1:3
    Lrgb(Lut_based_image(:)==i,i)=255;
end
Lrgb=reshape(Lrgb,[size(img_1) 3]);

axes(handles.axes15)
imshow(double(Lrgb))
global Umap
global selected_cluster_kmeans
Umap=fun_FM2map(img_1,fuzzy_U,fuzzy_H);

FuzzyCMeans_cluster1=uint8(Umap(:,:,1));
FuzzyCMeans_cluster2=uint8(Umap(:,:,2));
FuzzyCMeans_cluster3=uint8(Umap(:,:,3));

%% Saving Results to Table
% % %  Final Centroid for Fuzzy C Means Algorithm
Results_Fuzzy_Centroid_Final=table();
Results_Fuzzy_Centroid_Final.Cluster1=Fuzzy_Centroid_Final(:,1);
Results_Fuzzy_Centroid_Final.Cluster2=Fuzzy_Centroid_Final(:,2);
Results_Fuzzy_Centroid_Final.Cluster3=Fuzzy_Centroid_Final(:,3);
% % %  Final Centroid for Fuzzy C Means Algorithm

Results_Fuzzy_Centroid_Details=table();
Results_Fuzzy_Centroid_Details.Cluster1=fuzzy_centroid_details(:,1);
Results_Fuzzy_Centroid_Details.Cluster2=fuzzy_centroid_details(:,2);
Results_Fuzzy_Centroid_Details.Cluster3=fuzzy_centroid_details(:,3);
%% Plotting
% fcmeans_input_fig=figure('Name','Fuzzy C Means Clustering Results');
figure(3),
subplot(231),imshow(double(FuzzyCMeans_cluster1)),title('FC Means Clutser 1');
subplot(232),imshow(double(FuzzyCMeans_cluster2)),title('FC Means Clutser 2');
subplot(233),imshow(double(FuzzyCMeans_cluster3)),title('FC Means Clutser 3');

% If necessary, you can also unpack the membership functions to produce 
% membership maps
global selected_cluster_fc
%% Selecting Best Cluster
selected_cluster_fc=fun_find_best_cluster(FuzzyCMeans_cluster1,FuzzyCMeans_cluster2,FuzzyCMeans_cluster3,2);

%% Display Results
figure(3),
subplot(235),imshow(double(selected_cluster_fc)),title('Fuzzy C Selected Cluster');

axes(handles.axes16)
imshow(double(selected_cluster_fc))
fcmeans_check_status=1;
set(handles. pushbutton4,'Enable','on');
else
    fuzzycmeans_check_status=0;
end

if isequal(kmeans_check_status,0) && isequal(fuzzycmeans_check_status,0)

cla(handles.axes10);
cla(handles.axes11);
cla(handles.axes12);
cla(handles.axes15);
cla(handles.axes16);
cla(handles.axes17);
    warndlg('No Clustering Algorithm is Selected');
end

%%  Step 4: Label Every Pixel in the Image Using the Results from KMEANS
else
     warndlg('Input File not Provided');
end

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% Refine Function 
global selected_cluster_kmeans
global selected_cluster_fc
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% %  Check if Input image is provided and segmentation is done or not ??
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%% Displaying refined Image in Case of Check Box 1 is selected

if handles.checkbox1.Value
    post_processed_selected_cluster_kmeans = bwareaopen(selected_cluster_kmeans, 150);
    axes(handles.axes12)
    imshow(double(post_processed_selected_cluster_kmeans))
    set(handles. pushbutton2,'Enable','off');
    set(handles. pushbutton4,'Enable','off');

end    
%% Displaying refined Image in Case of Check Box 2 is selected

if handles.checkbox2.Value
    post_processed_selected_cluster_fc = bwareaopen(selected_cluster_fc, 150);
    axes(handles.axes17)
    imshow(double(post_processed_selected_cluster_fc))
    set(handles. pushbutton2,'Enable','off');
    set(handles. pushbutton4,'Enable','off');

end


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global I
% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject,'Color','None')


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(cancer)
first_page


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1
    set(handles. pushbutton2,'Enable','on');
    set(handles. pushbutton4,'Enable','on');


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles. pushbutton2,'Enable','on');
set(handles. pushbutton4,'Enable','on');

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on key press with focus on checkbox1 and none of its controls.
function checkbox1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
    set(handles. pushbutton2,'Enable','on');
    set(handles. pushbutton4,'Enable','on');


% --- Executes on key press with focus on checkbox2 and none of its controls.
function checkbox2_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
    set(handles. pushbutton2,'Enable','on');
    set(handles. pushbutton4,'Enable','on');
