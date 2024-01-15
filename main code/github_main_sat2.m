% Running Guidance: Local LQR -> calculate initial state -> integrate using
% parfor -> draw value function 1 & 2
% Or you can just press the "RUN" button
% Before running the main code, you need to first run the nonsmooth line
% code to get the nonsmooth line and the smooth region.
%% Local LQR problem
% m is the mass of pendulum, l is the length of pendulum ,b is the
% damping coefficient ,g is the gravity constant and umax is the control
% constrain
m = 1;
l = 1;
b = 0.1;
g = 9.8;
umax = 2;

% linear system: \dot x = Ax + Bu
B = [0;1/m/l^2];
A = [0,1;g/l,-b/m/l^2];
q = [1,1];
Q = [q(1),0;0,q(2)];
R = 1;

% do not change this
E = [1,0;0,1];
G = [0,0;0,0];
S = [0;0];
[P,~,~] = icare(A,B,Q,R,S,E,G);

[V,~] = eig(P);
K = eig(P);
disp("done LQR");
%% calculate initial state
% N is the num of trajectories, epsilon define the local LQR region, e is
% the initial state for trajectories, theta is the parameter of initial
% state on boundary of the LQR region(so actually e and theta are
% equivalent), Vc is the same as in the paper
N = 5000;
epsilon = 0.0002;
e = zeros(0,2);
%intialize theta
theta = [0,pi/2,pi,pi*3/2];
levelset = zeros(0,2);
Vc = 250000*epsilon;
figure;
for xi = 1 : N
    %xi is the index of trajectory, yi is the position
    yi = 0;
    if(xi<5)
        % initialize
        e1 = V(:,1)*cos(theta(xi))+V(:,2)*sin(theta(xi));
        yi = xi;
    else
        % calculating the distance between trajectories
        % the second part can avoid some numerical issue
        [~,index] = max(sum(abs(levelset-[levelset(2:end,:);levelset(1,:)]),2).*sum(abs(e-[e(2:end,:);e(1,:)]).^0.8,2));
        yi = index + 1;
        if(index == xi-1)
            if(abs(theta(end)-theta(1))<pi)
               theta(end+1) = (theta(end)+theta(1))/2;
            else
               theta(end+1) = (theta(end)+theta(1))/2 + pi;
            end
            e1 = V(:,1)*cos(theta(end))+V(:,2)*sin(theta(end));        
        else
            if(abs(theta(index)-theta(index+1))<pi)
               theta = [theta(1:index),(theta(index)+theta(index+1))/2,theta(index+1:end)];
            else
               theta = [theta(1:index),(theta(index)+theta(index+1))/2 + pi,theta(index+1:end)];
            end
            e1 = V(:,1)*cos(theta(index+1))+V(:,2)*sin(theta(index+1));
            levelset = [levelset(1:index,:);0,0;levelset(index+1:end,:)];
            e = [e(1:index,:);0,0;e(index+1:end,:)];
        end
    end
    % normalize it into the boundary of L
    % solve ODE
    e1 = sqrt(epsilon/(e1'*P*e1))*e1;
    e(yi,:) = e1;
    j = 2*P*e1;
    H1 = Vc;
    options = odeset('Reltol',3e-14,'AbsTol',3e-14,'Events',@(t,y)odeEventFun(t,y,H1));
    % options = odeset('MaxStep',0.001,'Reltol',1e-15,'AbsTol',1e-16,'Events',@(t,y)odeEventFun(t,y,H1));
    [~,y] = ode89(@(t,x)odefunc_pen(x,m,l,b,g,q,R,umax),[0 50],[e1(1) e1(2) j(1) j(2) e1'*P*e1],options);

    [~,indexl] = min(abs(y(:,5)-Vc));
    levelset(yi,:) = [y(indexl(1),1),y(indexl(1),2)];
    

    plot(y(indexl(1),1),y(indexl(1),2),'o','markers',2,'markerfacecolor','r');
    plot(y(:,1),y(:,2));
    hold on
end
xlim([-6,2]);
ylim([-4,4]);
disp("done initial state");
%% integrate using parfor
load("nonsmooth-sat2.mat");
points = zeros(0,5);
numdata = zeros(N,1);
% here using for is alos ok
parfor i = 1:N
    if(mod(i,100)==0)
        disp(i);
    end
    e1 = e(i,:)';
    j = 2*P*e1;
    H1 = 100;
    options = odeset('Reltol',3e-14,'AbsTol',1e-16,'Events',@(t,y)odeEventFun(t,y,H1));
    [ty,y] = ode89(@(t,x)odefunc_pen(x,m,l,b,g,q,R,umax),[0 50],[e1(1) e1(2) j(1) j(2) e1'*P*e1],options);
    [in,on] = inpolygon(y(:,1),y(:,2),bsp(:,1),bsp(:,2));
    nin = find(in ~= 1);
    if(size(nin,1)~=0)
        y((nin(1)):end,:) = [];
    end
    points = [points;y];
    numdata(i) = size(y,1);  
end
for i = 2:N
    numdata(i) = numdata(i-1) + numdata(i);
end
numdata = [0;numdata(1:end)];
points = points((points(:,5)<76.5 | abs(points(:,2))>5 | abs(points(:,1))>5),:);
disp("done trajectories");
%% draw value function 1
% N1 is the resolution of mesh,X,Y is x-axis and y-axis, bias is how we can
% + or - 2k\pi to move the point into the smooth region(in order to use
% interpolation)
% Actually we do not use the same amount of data as in paper, so this is
% not the best
N1 = 50;
epi = 0.1;
[X,Y] = meshgrid(8*(-1:1/N1:1),8*(-1:1/N1:1));

bias = zeros(2*N1+1,2*N1+1);
points_y = points(:,2);
[points_y_sort,points_y_sort_index] = sort(points_y);
for i = 1:2*N1+1
    for j = 1:2*N1+1
        x = X(i,j);
        y = Y(i,j);
        x1 = x;
        indexA = binarySearch(points_y_sort,y-epi);
        indexB = binarySearch(points_y_sort,y+epi);
        index = points_y_sort_index(indexA:indexB);
        xnear = sum(points(index,1))/size(index,1);
        while (abs(x1-xnear)>pi)
            bias(i,j) = bias(i,j) - (x1-xnear)/abs(x1-xnear)*2*pi;
            x1 = x1 - (x1-xnear)/abs(x1-xnear)*2*pi;
        end
        [in,on] = inpolygon(x1,y,bsp(:,1),bsp(:,2));
        if(in ~= 1)
           [in,on] = inpolygon(x1+2*pi,y,bsp(:,1),bsp(:,2));
           if(in == 1)
               bias(i,j) = bias(i,j) + 2*pi;
           else
               bias(i,j) = bias(i,j) - 2*pi;
           end
        end
    end
end
%% draw value function 2
N1 = 50;
epi = 0.1;
[X,Y] = meshgrid(8*(-1:1/N1:1),8*(-1:1/N1:1));
Z = 10*ones(2*N1+1,2*N1+1);
u1 = scatteredInterpolant(points(:,1),points(:,2),points(:,5),'natural');
% f1 = scatteredInterpolant(points(:,1),points(:,2),points(:,3),'natural');
% f2 = scatteredInterpolant(points(:,1),points(:,2),points(:,4),'natural');
h = [0.00001,0.0001*(1:9),0.001*(1:9),0.01*(1:9)];
for i = 1:2*N1+1
    for j = 1:2*N1+1
        x = X(i,j);
        y = Y(i,j);

        Z(i,j) = u1(x+bias(i,j),y);
        % DJ1(i,j) = f1(x+bias(i,j),y);
        % DJ2(i,j) = f2(x+bias(i,j),y);
        % for k = 1:size(h,2)
        %     J1 = u1(x+bias(i,j)+h(k),y);
        %     J2 = u1(x+bias(i,j)-h(k),y);
        %     J3 = u1(x+bias(i,j),y+h(k));
        %     J4 = u1(x+bias(i,j),y-h(k));
        %     J5 = u1(x+bias(i,j)+2*h(k),y);
        %     J6 = u1(x+bias(i,j)-2*h(k),y);
        %     J7 = u1(x+bias(i,j),y+2*h(k));
        %     J8 = u1(x+bias(i,j),y-2*h(k));
        %     dJ = [(J6-8*J2+8*J1-J5)/12/h(k);(J8-8*J4+8*J3-J7)/12/h(k)];
        %     u = sat(-dJ(2)/2/m/l^2/R,mu);
        %     re = q(1)*(2-2*cos(x))+q(2)*y^2+R*u^2+dJ(1)*y+dJ(2)*(-y*b/m/l^2+g/l*sin(x)+u/m/l^2);
        %     Z(i,j) = min(sat(log10(abs(re)),8),Z(i,j));
        %     % if(abs(re)<RE(i,j))
        %     %     Z1(i,j) = sat(dJ(1),50);
        %     %     Z2(i,j) = sat(dJ(2),50);
        %     %     RE(i,j) = abs(re);
        %     % end
        %     % Z(i,j) = min(sat(abs(dJ(2)-F2(x+bias(i,j),y)),8),Z(i,j));
        % end
    end
end

figure;
contourf(X,Y,Z,10);
contourcbar
axis equal
figure;
s = surf(X,Y,Z);
set(gca,'FontSize',35);
disp("Finished!");
%% function
function dxdt = odefunc_pen(x,m,l,b,g,q,R,umax)
    dxdt = zeros(5,1);
    x1 = x(1);
    x2 = x(2);
    j1 = x(3);
    j2 = x(4);
    u = -j2/2/m/l^2/R;
    u = sat(u,umax);
    dxdt(1) = -x2;
    dxdt(2) = -(-x2*b/m/l^2+g/l*sin(x1)+u/m/l^2);
    dxdt(3) = -(-2*q(1)*sin(x1)-g/l*cos(x1)*j2);
    dxdt(4) = -(-2*q(2)*x2-j1+b/m/l^2*j2);
    dxdt(5) = q(1)*(2-2*cos(x1))+q(2)*x2^2+R*(u)^2;
end

function dxdt = odefunc_pen_control_min(x,m,l,b,g,R,F,mu,bsp1,bias)
    dxdt = zeros(3,1);
    x1 = x(1);
    x2 = x(2); 
    u = -F(x1+bias,x2)/2/m/l^2/R;
    u = sat(u,mu);
    dxdt(1) = x2;
    dxdt(2) = (-x2*b/m/l^2+g/l*sin(x1)+u/m/l^2);
    dxdt(3) = 2*(1-cos(x1))+x2^2+u^2;
end

function [value,isterminal,direction] = odeEventFun(t,y,H1)
    value=y(5)-H1; %test if done
    isterminal=1;   
    direction=1;     
end

function uo = sat(u,um)
    if(u>um)
        uo = um;
    elseif(u<-um)
        uo = -um;
    else
        uo = u;
    end
end

function updatePlot_i(src, event, cLine1 ,cLine2 ,cLine3 ,points,numdata,N)
    newAmplitude = get(src, 'Value');
    cont = zeros(N,2);
    for i = 1:N
        [~,index] = min(abs(points(numdata(i)+1:numdata(i+1),5)-newAmplitude));
        cont(i,:) = [points(numdata(i)+index(1),1),points(numdata(i)+index(1),2)];
    end
    set(cLine1, 'XData', cont(:,1),'YData', cont(:,2)); 
    set(cLine2, 'XData', cont(:,1)+2*pi,'YData', cont(:,2)); 
    set(cLine3, 'XData', cont(:,1)-2*pi,'YData', cont(:,2)); 
    title(['contour：', num2str(newAmplitude)]); 
    xlim([-10,10]);
    ylim([-10,10]);
end

function updatePlot(src, event, cLine1,points,numdata,sequ,N)
    newAmplitude = get(src, 'Value');
    cont = zeros(N,2);
    for i = 1:N
        [~,index] = min(abs(points(numdata(sequ(i))+1:numdata(sequ(i)+1),5)-newAmplitude));
        cont(i,:) = [points(numdata(sequ(i))+index(1),1),points(numdata(sequ(i))+index(1),2)];
    end
    set(cLine1, 'XData', cont(:,1),'YData', cont(:,2)); 
    title(['contour：', num2str(newAmplitude)]); 
    xlim([-10,10]);
    ylim([-10,10]);
end

function index = binarySearch(arr, target)
    left = 1;       
    right = length(arr);  
    while left <= right
        mid = floor((left + right) / 2);    
        if arr(mid) == target
            index = mid; 
            return
        elseif arr(mid) < target
            left = mid + 1;    
        else
            right = mid - 1;   
        end
    end
    index = mid;
     
end