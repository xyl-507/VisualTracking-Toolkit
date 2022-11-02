% copy from��https://blog.csdn.net/weixin_45562620/article/details/123906438
% https://blog.csdn.net/qq_44823829/article/details/109380603
%  https://blog.csdn.net/juanji3798/article/details/119013621

% ��������ռ�����б�������������MEX�ļ�
clear all; clc;

% ���������ļ���������ΪA,B,GT
A = load('bike1_AutoTrack.txt');  %UAV123�ı�ע��ʽ�����Ͻ�+��ߣ�x y w h
B = load('bike1_SiamRPN++.txt');
GT = load('bike1.txt');
% ����A�Ĺ�ģ��[��,��]
[m,n] = size(A);

%% ���彻����
iou1 = zeros(m,1);
iou2 = zeros(m,1);
cle_1 = zeros(m,1);
cle_2 = zeros(m,1);
% ���м���iou,cle
for q = 1:m
    % ���ϽǺͿ��
    x_1=A(q,1);y_1=A(q,2);w_1=A(q,3);h_1=A(q,4);
    x_2=B(q,1);y_2=B(q,2);w_2=B(q,3);h_2=B(q,4);
    x_gt=GT(q,1);y_gt=GT(q,2);w_gt=GT(q,3);h_gt=GT(q,4);
    % ���ĵ�����
    cx_1=x_1+w_1/2; cy_1=y_1-h_1/2;
    cx_2=x_2+w_2/2; cy_2=y_2-h_2/2;
    cx_gt=x_gt+w_gt/2; cy_gt=y_gt-h_gt/2;

% �����㷨1��IoU
    % �����ص����ֵĿ�͸�
    endx13 = max(x_1 + w_1, x_gt + w_gt);
    startx13 = min(x_1, x_gt);
    w13 = w_1 + w_gt - (endx13 - startx13);
 
    endy13 = max(y_1 + h_1, y_gt + h_gt);
    starty13 = min(y_1, y_gt);
    h13 = h_1 + h_gt - (endy13 - starty13);
    
    % ����ص�����Ϊ��, �����ص�
    if w13 <= 0 || h13 <= 0
        iou1(q,1) = 0;
    else
        Area13 = w13 * h13;
        Area13_1 = w_1 * h_1;
        Area13_2 = w_gt * h_gt;
        iou1(q,1) = Area13 * 1.0 / (Area13_1 + Area13_2 - Area13);
    end
    
% �����㷨2��IoU
    % �����ص����ֵĿ�͸�
    endx23 = max(x_2 + w_2, x_gt + w_gt);
    startx23 = min(x_2, x_gt);
    w23 = w_2 + w_gt - (endx23 - startx23);
 
    endy23 = max(y_2 + h_2, y_gt + h_gt);
    starty23 = min(y_2, y_gt);
    h23 = h_2 + h_gt - (endy23 - starty23);
 
    % ����ص�����Ϊ��, �����ص�
    if w23 <= 0 || h23 <= 0
        iou2(q,1) = 0;
    else
        Area23 = w23 * h23;
        Area23_1 = w_2 * h_2;
        Area23_2 = w_gt * h_gt;
        iou2(q,1) = Area23 * 1.0 / (Area23_1 + Area23_2 - Area23);
    end
    % �������cle
%     cle_1(q,1) = sqrt((cx_1 - cx_gt).^2+(cy_1 - cy_gt).^2);
%     cle_2(q,1) = sqrt((cx_2 - cx_gt).^2+(cy_2 - cy_gt).^2);
    cle_1(q,1) = sqrt(sum(([cx_1,cy_1]-[cx_gt,cy_gt]).^2));
    cle_2(q,1) = sqrt(sum(([cx_2,cy_2]-[cx_gt,cy_gt]).^2));
end
%% iou
% ����txt�ļ���һ�е�����
figure(1);
plot(iou1(:,1));
hold on
plot(iou2(:,1));
% ������
xlabel('Frame');
% ������
ylabel('IoU');
% ͼ��
legend('SiamRPN++ [0.765]','AutoTrack [0.345]')
% ����
title('IoU curve of sequence bike1');
%% cle
figure(2);
plot(cle_1(:,1));
hold on
plot(cle_2(:,1));
% ����x=1��ֱ��
plot([0,m],[20,20],'m--','LineWidth',1.2); %m-- �����������ͣ���ɫ������

% ������
xlabel('Frame');
% ������
ylabel('CLE');
legend('SiamRPN++','AutoTrack')
% ����
title('CLE curve of sequence bike1');
set(gca, 'yTick', [0,20,50:50:150]);
set(gca,'YTickLabel',{'0','20','50','100','150'})% ͼ��
