function [Centroid_Final,Fuzzy_Membership,LUT,H,Centroid_Track,Distance_to_Centroid]=FastFCMeans(im,c,q,opt)

% Default input arguments
if nargin<2 || isempty(c), c=2; end
if nargin<3 || isempty(q), q=2; end
if nargin<4 || isempty(opt), opt=true; end

% Basic error checking
if nargin<1 || isempty(im)
    error('Insufficient number of input arguments')
end
msg='Revise variable used to specify class centroids. See function documentation for more info.';
if ~isnumeric(c) || ~isvector(c)
    error(msg)
end
if numel(c)==1 && (~isnumeric(c) || round(c)~=c || c<2)
    error(msg)
end
if ~isnumeric(q) || numel(q)~=1 || q<1.1
    error('3rd input argument (q) must be a real number > 1.1')
end
if ~islogical(opt) || numel(opt)>1
    error('4th input argument (opt) must a Boolean')
end    

% Check image format
if isempty(strfind(class(im),'int'))
    error('Input image must be specified in integer format (e.g. uint8, int16)')
end

% Intensity range
Imin=double(min(im(:)));
Imax=double(max(im(:)));
I=(Imin:Imax)';

% Initialize cluster centroids
if numel(c)>1 % user-defined centroids
    Centroid_Final=c;
    opt=true;
else % automatic initialization
    if opt
        dI=(Imax-Imin)/c;
        Centroid_Final=Imin+dI/2:dI:Imax;
    else
        [Centroid_Final,~,H]=FastCMeans(im,c);
    end
end

%% Compute intensity histogram
if opt
    H=hist(double(im(:)),I);
    H=H(:);
end
clear im

%% Update fuzzy memberships and cluster centroids
dC=Inf;
start=1;
while dC>1E-3
    
    C0=Centroid_Final;
    
    %% Distance to the centroids
    Distance_to_Centroid=abs(bsxfun(@minus,I,Centroid_Final));
    Distance_to_Centroid=Distance_to_Centroid.^(2/(q-1))+eps;
    
    %% Compute fuzzy memberships
    Fuzzy_Membership=bsxfun(@times,Distance_to_Centroid,sum(1./Distance_to_Centroid,2));
    Fuzzy_Membership=1./(Fuzzy_Membership+eps);
    
    %% Update the centroids
    UH=bsxfun(@times,Fuzzy_Membership.^q,H);
    Centroid_Final=sum(bsxfun(@times,UH,I),1)./sum(UH,1);
    Centroid_Final=sort(Centroid_Final,'ascend'); % enforce natural order
    Centroid_Track(start,:)=Centroid_Final;
    % Change in centroids 
    dC=max(abs(Centroid_Final-C0));
    
    start=start+1;
end

% Defuzzify
[~,LUT]=max(Fuzzy_Membership,[],2);

