function Umap=FM2map(im,U,H)
% Unpack fuzzy-membership functions to produce membership maps.
% 
% INPUT:
%   - im    : N-dimensional grayscale image in integer format. 
%   - U     : L-by-c array of fuzzy class memberships, where c is the 
%             number of classes and L is the intensity range of the input 
%             image, such that L=numel(min(im(:)):max(im(:))). See 
%             'FastFCMeans' function for more info.
%   - H     : image histogram returned by 'FastFCMeans' function.
%
% OUTPUT:
%   - Umap  : membership maps. Umap has the same size as the input image
%             plus an additional dimension to account for c classes. For
%             example, if im is a 2D M-by-N image then U will be 
%             M-by-N-by-c array where U(:,:,i) is a membership map for the
%             i-th class.
%
% AUTHOR    : Anton Semechko (a.semechko@gmail.com)
%


if nargin<3 || isempty(H)

    % Intensity range
    Imin=double(min(im(:)));
    Imax=double(max(im(:)));
    I=(Imin:Imax)';
    
    % Intensity histogram
    H=hist(double(im(:)),I);
    H=H(:);

end

% Unpack memberships
Umap=zeros(sum(H),size(U,2));
i1=1; i2=0;
for i=1:numel(H)
    i2=i2+H(i);
    Umap(i1:i2,:)=repmat(U(i,:),[H(i) 1]);
    i1=i2+1;
end

% Find positional mapping
[~,idx]=sort(im(:), 'ascend');
[~,idx]=sort(idx(:),'ascend');

% Reshape membership maps to match image dimensions
Umap=reshape(Umap(idx,:),[size(im) size(U,2)]);

