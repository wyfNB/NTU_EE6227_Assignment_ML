function pred_label = Bayes_decesion_rule(Data_Train,Label_Train,test)
n1 = 0;
n2 = 0;
n3 = 0;
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

u1_hat = sum(class_1,1)/n1;
u2_hat = sum(class_2,1)/n2;
u3_hat = sum(class_3,1)/n3;

sigma_1_hat = zeros(size(Data_Train,2),size(Data_Train,2));
for i = 1:n1
    sigma_1_hat = sigma_1_hat + (class_1(i,:)-u1_hat)'*(class_1(i,:)-u1_hat);
end
sigma_1_hat = sigma_1_hat/n1;

sigma_2_hat = zeros(size(Data_Train,2),size(Data_Train,2));

for i = 1:n2
    sigma_2_hat = sigma_2_hat + (class_2(i,:)-u2_hat)'*(class_2(i,:)-u2_hat);
end
sigma_2_hat = sigma_2_hat/n2;

sigma_3_hat = zeros(size(Data_Train,2),size(Data_Train,2));

for i = 1:n3
    sigma_3_hat = sigma_3_hat + (class_3(i,:)-u3_hat)'*(class_3(i,:)-u3_hat);
end
sigma_3_hat = sigma_3_hat/n3;
d = size(Data_Train,2);
pred_label = zeros(size(test,1),1);
for i = 1:size(test,1)
    g1 = 0.5*(1/((2*pi)^(d)*sqrt(det(sigma_1_hat))))*exp(-0.5*(test(i,:)-u1_hat)/sigma_1_hat*(test(i,:)-u1_hat)');
    g2 = 0.5*(1/((2*pi)^(d)*sqrt(det(sigma_2_hat))))*exp(-0.5*(test(i,:)-u2_hat)/sigma_2_hat*(test(i,:)-u2_hat)');
    g3 = 0.5*(1/((2*pi)^(d)*sqrt(det(sigma_3_hat))))*exp(-0.5*(test(i,:)-u3_hat)/sigma_3_hat*(test(i,:)-u3_hat)');
    [max_value,max_index] = max([g1,g2,g3]);
    pred_label(i) = max_index;
end