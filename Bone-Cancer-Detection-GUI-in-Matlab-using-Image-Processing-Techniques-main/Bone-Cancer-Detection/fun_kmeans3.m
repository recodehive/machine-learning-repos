function [gIdx,cluster_center,z2]=fun_kmeans3(X,k)
%   K_MEANS    Improved k-means clustring
%   IDX = k_means(X, K) partititions the N x P data matrix X into K
%   clusters through a fully vectorized algorithm, where N is the number of
%   data points and P is the number of dimensions (variables). The
%   partition minimizes the sum of point-to-cluster-centroid Euclidean
%   distances of all clusters. The returned N x 1 vector IDX contains the
%   cluster indices of each point.
%
%   IDX = k_means(X, C) works with the initial centroids, C, (K x P).
%
%   [IDX, C] = k_means(X, K) also returns the K cluster centroid locations
%   in the K x P matrix, C.
%
% See also kmeans

% Version 2.0, by Yi Cao at Cranfield University on 27 March 2008.

% Check input and output
% error(nargchk(2,3,nargin));
% error(nargoutchk(0,3,nargout));

[n,m]=size(X);

% Check if second input is centroids
if ~isscalar(k)
    cluster_center=k;
    k=size(cluster_center,1);
else
    cluster_center=X(ceil(rand(k,1)*n),:);
end

% allocating variables
g0=ones(n,1);
gIdx=zeros(n,1);
D=zeros(n,k);

% Main loop converge if previous partition is the same as current
while any(g0~=gIdx)
%     disp(sum(g0~=gIdx))
    g0=gIdx;
    % Loop for each centroid
    for t=1:k
        d=zeros(n,1);
        % Loop for each dimension
        for s=1:m
            d=d+(X(:,s)-cluster_center(t,s)).^2;
        end
        D(:,t)=d;
    end
    % Partition data to closest centroids
    [z,gIdx]=min(D,[],2);
   [z2,cluster_index]=min(D,[],3);
    
    % Update centroids using means of partitions
    for t=1:k
        cluster_center(t,:)=mean(X(gIdx==t,:));
        
    end
%     for t=1:m
%         c(:,t)=accumarray(gIdx,X(:,t),[],@mean);
%     end
end