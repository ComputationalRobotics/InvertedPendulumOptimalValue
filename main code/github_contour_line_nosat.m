% Running Guidance: Local LQR -> calculate initial state ->
% integrate using parfor -> calculate non-smooth line -> do more on the
% data -> save non-smooth line
% Or you can just press the "RUN" button
%% Local LQR problem
% m is the mass of pendulum, l is the length of pendulum ,b is the
% damping coefficient and g is the gravity constant
m = 1;
l = 1;
b = 0.1;
g = 9.8;

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
% equivalent), Vc is the same as in the paper.
N = 2000;
epsilon = 0.0002;
e = zeros(0,2);
%intialize theta
theta = [0,pi/2,pi,pi*3/2];
levelset = zeros(0,2);
Vc = 100000*epsilon;
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
        [~,index] = max(sum(abs(levelset-[levelset(2:end,:);levelset(1,:)]),2).*sum(abs(e-[e(2:end,:);e(1,:)]).^0.5,2));
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
    e1 = sqrt(epsilon/(e1'*P*e1))*e1;
    e(yi,:) = e1;
    j = 2*P*e1;
    H1 = 100;
    options = odeset('Reltol',3e-14,'AbsTol',3e-14,'Events',@(t,y)odeEventFun(t,y,H1));
    % options = odeset('MaxStep',0.001,'Reltol',1e-15,'AbsTol',1e-16,'Events',@(t,y)odeEventFun(t,y,H1));
    % solve ODE
    [~,y] = ode89(@(t,x)odefunc_pen(x,m,l,b,g,q,R),[0 50],[e1(1) e1(2) j(1) j(2) e1'*P*e1],options);

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
points = zeros(0,5);
numdata = zeros(N,1);
% here using for is alos ok
parfor i = 1:N
    if(mod(i,500)==0)
        disp(i);
    end
    e1 = e(i,:)';
    j = 2*P*e1;
    H1 = 100;
    options = odeset('maxstep',0.005,'Reltol',3e-14,'AbsTol',1e-16,'Events',@(t,y)odeEventFun(t,y,H1));
    [ty,y] = ode89(@(t,x)odefunc_pen(x,m,l,b,g,q,R),[0 50],[e1(1) e1(2) j(1) j(2) e1'*P*e1],options);
    points = [points;y];
    numdata(i) = size(y,1);  
end
for i = 2:N
    numdata(i) = numdata(i-1) + numdata(i);
end
numdata = [0;numdata(1:end)];
points_save = points;
disp("done trajectories");
%% figure contour line
% this part is about the contour line as a figure
Vc_v = 20;
figure;
cont = zeros(N,2);
for i = 1:N
    [~,index] = min(abs(points(numdata(i)+1:numdata(i+1),5)-Vc_v));
    cont(i,:) = [points(numdata(i)+index(1),1),points(numdata(i)+index(1),2)];
end
cLine1 = plot(cont(:,1),cont(:,2));
hold on
cLine2 = plot(cont(:,1)+2*pi,cont(:,2));
hold on
cLine3 = plot(cont(:,1)-2*pi,cont(:,2));
hold on
xlim([-8,8]);
ylim([-8,8]);
slider = uicontrol('Style', 'slider', 'Min', 20, 'Max', 120, 'Value', 20, 'Position', [150 10 400 50]);
addlistener(slider, 'ContinuousValueChange', @(src, event)updatePlot_i(src, event, cLine1 ,cLine2 ,cLine3 ,points,numdata,N));
%% calculate non-smooth line
% delta and num of values
Vc_v = 0:0.1:80;
figure;
cont = zeros(N,2);
% the insert point part is designed for discontinuous part, because the
% contour line will disappear near the discontinuous line, so we need to
% record it.
insert_index = [];
insert_points = {};
% record the non-smooth line
boundary_spiral = zeros(4*size(Vc_v,2),2);
onpoints = zeros(4,2);
for k = 1:size(Vc_v,2)
    for i = 1:N
        [~,index] = min(abs(points(numdata(i)+1:numdata(i+1),5)-Vc_v(k)));
        cont(i,:) = [points(numdata(i)+index(1),1),points(numdata(i)+index(1),2)];
        
    end

    % find these points who is on the discontinuous line and record it
    cont_dist = sum((cont-[cont(2:end,:);cont(1,:)]).^2,2);
    vb = find(cont_dist>4);
    % update the insert points
    for j = 1:size(vb,1)
        vf = find(insert_index == vb(j));
        if(isempty(vf) && size(insert_index,2)<4)
            insert_index(end+1) = vb(j);
            insert_points{end+1} = [cont(vb(j),:);cont(mod(vb(j),size(cont,1))+1,:)];
        elseif(~isempty(vf))
            insert_points{vf} = [cont(vb(j),:);insert_points{vf};cont(mod(vb(j),size(cont,1))+1,:)];
        end
    end 
    cont_final = cont;
    [~,vI] = sort(insert_index,'descend');
    for i = 1:size(vI,2)
        cont_final = [cont_final(1:insert_index(vI(i)),:);insert_points{vI(i)};cont_final((insert_index(vI(i))+1):end,:)];
    end

    poly1 = polyshape([cont_final(:,1),cont_final(:,2)]);
    poly2 = polyshape([cont_final(:,1)+2*pi,cont_final(:,2)]);
    [polyout,SID,VID] = intersect(poly1,poly2);

    %update intersection point
    if(size(polyout.Vertices,1)~=0)
        ind = find(SID==0);
        if(size(ind,1)==4)
           scatter(polyout.Vertices(ind,1),polyout.Vertices(ind,2));
           boundary_spiral((4*(k-1)+1):4*k,:) = polyout.Vertices(ind,:);
           onpoints = [polyout.Vertices(ind,1),polyout.Vertices(ind,2)];
           hold on 
        elseif(size(ind,1)>4)
            I = zeros(4,1);
            for j = 1:4
                [~,v] = min((polyout.Vertices(ind(:),1)-onpoints(j,1)).^2+(polyout.Vertices(ind(:),2)-onpoints(j,2)).^2);
                I(j) = v;
            end
            scatter(polyout.Vertices(ind(I),1),polyout.Vertices(ind(I),2));
            boundary_spiral((4*(k-1)+1):4*k,:) = polyout.Vertices(ind(I),:);
            onpoints = [polyout.Vertices(ind(I),1),polyout.Vertices(ind(I),2)];
            hold on
        end
    end
end
%% do more on the data
% one thing important is as the parameter(m,l,g) changes, the spiral line
% may have some other changes, you should refine the non-smooth line
% yourself.
boundary_spiral(all(boundary_spiral==0,2),:)=[];
% if you change the parameter you should change this one
boundary_spiral = boundary_spiral(1:1252,:);
a_bsp = boundary_spiral(4*(1:size(boundary_spiral,1)/4-1),:);
c_bsp = boundary_spiral(4*(1:size(boundary_spiral,1)/4-1)+2,:);
bsp = [flip([2*pi-c_bsp(:,1),-c_bsp(:,2)]);a_bsp];
bsp = [bsp;flip([2*pi-bsp(:,1),-bsp(:,2)])];
bsp = [bsp;flip([bsp(:,1)-2*pi,bsp(:,2)])];
plot(polyshape(bsp(:,1),bsp(:,2)));
disp("done nonsmooth line");
%% save non-smooth line
save('nonsmooth-nosat.mat',"bsp");
disp("Finished!");
%% function
function dxdt = odefunc_pen(x,m,l,b,g,q,R)
    dxdt = zeros(5,1);
    x1 = x(1);
    x2 = x(2);
    j1 = x(3);
    j2 = x(4);
    u = -j2/2/m/l^2/R;

    dxdt(1) = -x2;
    dxdt(2) = -(-x2*b/m/l^2+g/l*sin(x1)+u/m/l^2);
    dxdt(3) = -(-2*q(1)*sin(x1)-g/l*cos(x1)*j2);
    dxdt(4) = -(-2*q(2)*x2-j1+b/m/l^2*j2);
    dxdt(5) = q(1)*(2-2*cos(x1))+q(2)*x2^2+R*(u)^2;
end

function dxdt = odefunc_pen_control(t,x,m,l,b,g,R,F2,mu)
    dxdt = zeros(2,1);
    x1 = x(1);
    x2 = x(2);  
    j2 = F2(x1,x2);
    u = -j2/2/m/l^2/R;
    u = sat(u,mu);
    dxdt(1) = x2;
    dxdt(2) = (-x2*b/m/l^2+g/l*sin(x1)+u/m/l^2);
end

function dxdt = odefunc_pen_control_min(x,m,l,b,g,R,F,mu,bias)
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