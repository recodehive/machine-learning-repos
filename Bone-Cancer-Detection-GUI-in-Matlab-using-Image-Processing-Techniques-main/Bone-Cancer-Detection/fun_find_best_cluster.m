function selected_cluster=fun_find_best_cluster(cluster1,cluster2,cluster3,type)

sum_cluster1=sum(cluster1(:));
sum_cluster2=sum(cluster2(:));
sum_cluster3=sum(cluster3(:));

if type==1
Final=max([sum_cluster1,sum_cluster2,sum_cluster3]);
elseif type==2
Final=min([sum_cluster1,sum_cluster2,sum_cluster3]);
end



    if isequal(Final,sum_cluster1)
    selected_cluster=cluster1;
%     disp('Cluster 1 is Selected');
    elseif isequal(Final,sum_cluster2)
        selected_cluster=cluster2;
%         disp('Cluster 2 is Selected');
    elseif isequal(Final,sum_cluster3)
        selected_cluster=cluster3;
%         disp('Cluster 3 is Selected');
    else
    end
end