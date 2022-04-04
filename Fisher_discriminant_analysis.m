function pred_label = Fisher_discriminant_analysis(Data_Train,Label_Train,test)
n1 = 0;
n2 = 0;
n3 = 0;
d = size(Data_Train,2);
for i = 1:size(Label_Train,1)
    if Label_Train(i) == 1
        n1 = n1+1;
    elseif Label_Train(i) == 2
        n2 = n2+1;
    else
        n3 = n3+1;
    end
end
class_1 = zeros(n1,4);
class_2 = zeros(n2,4);
class_3 = zeros(n3,4);

j = 1;
k = 1;
l = 1;
for i =1:size(Label_Train,1)
    if Label_Train(i) == 1
        class_1(j,:) = Data_Train(i,:);
        j = j+1;
    elseif Label_Train(i) == 2
        class_2(k,:) = Data_Train(i,:);
        k = k+1;
    else
        class_3(l,:) = Data_Train(i,:);
        l = l+1;        
    end
end

m1 = sum(class_1,1)/n1;
m2 = sum(class_2,1)/n2;
m3 = sum(class_3,1)/n3;
m = sum(Data_Train,1)/size(Data_Train,1);

s1 = zeros(d,d);
for i = 1:n1
    s1 = s1 + (class_1(i,:)-m1)'*(class_1(i,:)-m1);
end

s2 = zeros(d,d);
for i = 1:n2
    s2 = s2 + (class_2(i,:)-m2)'*(class_2(i,:)-m2);
end

s3 = zeros(d,d);
for i = 1:n3
    s3 = s3 + (class_3(i,:)-m3)'*(class_3(i,:)-m3);
end
s_w = s1+s2+s3;

s_t = zeros(d,d);
for i = 1:size(Data_Train,1)
    s_t = s_t + (Data_Train(i,:)-m)'*(Data_Train(i,:)-m);
end

s_b = s_t - s_w;

[eigen_vector,~]=eig(s_w\s_b);
eigen_vector = real(eigen_vector);
eigen_value = eig(s_w\s_b);
eigen_value = real(eigen_value);
[~,index] = sort(eigen_value(:),'descend');

w_1 = eigen_vector(:,index(1));
w_2 = eigen_vector(:,index(2));

%calculate w01
m11 = 0;
for i=1:n1
    m11 = m11 + w_1'*class_1(i,:)';
end
m11 = m11/n1;

m12 = 0;
for i=1:n2
    m12 = m12 + w_1'*class_2(i,:)';
end
m12 = m12/n2;

m13 = 0;
for i=1:n3
    m13 = m13 + w_1'*class_3(i,:)';
end
m13 = m13/n3;

[~,distance_index] = max([abs(m11-m12)+abs(m11-m13),abs(m12-m11)+abs(m12-m13),abs(m13-m11)+abs(m13-m12)]);
mii = [abs(m11-m12),abs(m11-m13);
       abs(m12-m11),abs(m12-m13);
       abs(m13-m11),abs(m13-m12)];
[~,closest_index] = min(mii(distance_index,:));
w_01_ii = [m11+m12,m11+m13;
           m12+m11,m12+m13;
           m13+m11,m13+m12];
w_01 = - w_01_ii(distance_index,closest_index)/2;

%calculate w02
m21 = 0;
for i=1:n1
    m21 = m21 + w_2'*class_1(i,:)';
end
m21 = m21/n1;

m22 = 0;
for i=1:n2
    m22 = m22 + w_2'*class_2(i,:)';
end
m22 = m22/n2;

m23 = 0;
for i=1:n3
    m23 = m23 + w_2'*class_3(i,:)';
end
m23 = m23/n3;

[~,distance_index2] = max([abs(m21-m22)+abs(m21-m23),abs(m22-m21)+abs(m22-m23),abs(m23-m21)+abs(m23-m22)]);
mii2 = [abs(m21-m22),abs(m21-m23);
       abs(m22-m21),abs(m22-m23);
       abs(m23-m21),abs(m23-m22)];
[~,closest_index2] = min(mii2(distance_index2,:));
w_02_ii = [m21+m22,m21+m23;
           m22+m21,m22+m23;
           m23+m21,m23+m22];
w_02 = - w_02_ii(distance_index2,closest_index2)/2;

class1 = distance_index;
class2 = distance_index2;
class3 = 6-class1-class2;

pred_label =zeros(size(test,1),1);
for i = 1:size(test,1)
    %calculate g1 g2
    g1 = w_1'*test(i,:)'+w_01;
    g2 = w_2'*test(i,:)'+w_02;

    %classification
    if (g1 >= 0) && (g2<=0)
        pred_label(i) = class1;
    elseif (g2>=0) && (g1<=0)
        pred_label(i) = class2;
    elseif (g1<=0) && (g2<=0)
        pred_label(i) = class3;
    else
        if g1>=g2
            pred_label(i) = class1;
        else
            pred_label(i) = class2;
        end
    end
end
end