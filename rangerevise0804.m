clear all;
clc;
%考虑每个方向通行情况，车多，车少以及堵车
t=10; %inner loop iteration time
B2_0=10;
P2_0=10;
P3_0=10;
N0=10^(-14.4);
Bm=150;
Pm=300;
% d12_R=1000*(0.002+0.029*rand(1,1));
% d23_V=(0.005+0.05*rand(1,1))*1000;
ga=[1.59,1.68,1.77,2,2.5,3,4];
pB=10;
pP=1;
P2_1=10;
alpha=1;
eps=1e-14;
eps2=1e-4;
M=100;%op的uf乘数
c0=1;
% c0=1e-8;
rho=5000;
halflen = 70;
%不同交通状况下的车距
% cardis_jam = 2;
% carmindis_jam = 4;
% cardis_heavy = 10;
% carmindis_heavy = 6;
% cardis_light = 50;
% carmindis_light = 16;
cardis=[2,10,50];
carmindis=[4,6,16];
coverrange = 40:4:60;
verd = [7,3,-3,-7];
r=zeros(4,35);
for i =1:4
    for j=1:35
        r(i,j)=10+2*rand(1,1);
    end
end
fxstar=zeros(500,9,7);
fxbefore=zeros(9,7);
totalcar=zeros(1,9);
crvcar=zeros(1,9);
% for rdir1con=1:3
%     for rdir2con=rdir1con:3
%         disp(rdir1con);
%         disp(rdir2con);
rdir1con=2;
rdir2con=3;
 %% car generating
            car_location = zeros(4,35);
            car_id = randperm(1000);%1-1000随机排列
            car_idnum = zeros(4,35);%存放每个位置的车序号
            car_bucket=cell(10,20);%10个桶，根据车序号个位数分配桶
            bucket_cnt=zeros(10,1);%桶内车辆计数
            idcnt=1;
            for i = 1:2%每条车道第一辆车
                car_location(i,1)=rand(1,1)*carmindis(rdir1con)-halflen;
                car_idnum(i,1)=car_id(idcnt);
                idcnt=idcnt+1;
            end
            for i = 3:4%每条车道第一辆车
                car_location(i,1)=rand(1,1)*carmindis(rdir2con)-halflen;
                car_idnum(i,1)=car_id(idcnt);
                idcnt=idcnt+1;
            end
            carnum = zeros(4,1);
            for i = 1:2
                for j = 2:35
                    car_location(i,j)=car_location(i,j-1)+carmindis(rdir1con)+rand(1,1)*cardis(rdir1con);
                    if car_location(i,j) > halflen
                        car_location(i,j)=NaN;
                        carnum(i) = j-1;%每条道车的数量
                        break;
                    end
                    car_idnum(i,j) = car_id(idcnt);
                    idcnt=idcnt+1;
                end
            end
            for i = 3:4
                for j = 2:35
                    car_location(i,j)=car_location(i,j-1)+carmindis(rdir2con)+rand(1,1)*cardis(rdir2con);
                    if car_location(i,j) > halflen
                        car_location(i,j)=NaN;
                        carnum(i) = j-1;%每条道车的数量
                        break;
                    end
                    car_idnum(i,j) = car_id(idcnt);
                    idcnt=idcnt+1;
                    
                end
            end
for rsel=1:6 
% rsel=2;
           
            crv=zeros(4,35);
            
            d12_R = zeros(4,35);
            d23_V = zeros(4,35);
            
            crvnum=0;
            for i = 1:2
                for j = 1:carnum(i)
                    if car_location(i,j)^2 + verd(i)^2 <= coverrange(rsel)^2
                        d12_R(i,j) = sqrt(car_location(i,j)^2 + verd(i)^2);
                        crvnum=crvnum+1;
                    end
                end 
            end
            for i = 3:4
                for j = 1:carnum(i)
                    if car_location(i,j)^2 + verd(i)^2 <= coverrange(rsel)^2
                        d12_R(i,j) = sqrt(car_location(i,j)^2 + verd(i)^2);
                        crvnum=crvnum+1;
                    end
                end
            end
            totalcar(rsel)=sum(carnum);
            crvcar(rsel)=crvnum;
%                     totalcar(gasel)=sum(carnum);
%                     crvcar(gasel)=crvnum;
            urvnum=sum(carnum)-crvnum;
            urvcellcnt=0;
            % relay generation
            CRV=cell(1,urvnum);
            URV=cell(1,urvnum);
            for i=1:4
                dir=0;
                urvrelaysel1=0;
                urvrelaysel2=0;
                for j = 1:carnum(i)
                    if d12_R(i,j)>0
                        dir=1;
                    else
                        if dir==0
                           urvrelaysel1=urvrelaysel1+1;
                        else
                            urvrelaysel2=urvrelaysel2+1;
                        end
                    end
                end
                for relaygen1=1:urvrelaysel1
                    URV{urvcellcnt+relaygen1}=[i,relaygen1];
                    CRV{urvcellcnt+relaygen1}=[i,relaygen1+urvrelaysel1];
                end
                urvcellcnt=urvcellcnt+urvrelaysel1;
                for relaygen2=1:urvrelaysel2
                    URV{urvcellcnt+relaygen2}=[i,carnum(i)-urvrelaysel2+relaygen2];
                    CRV{urvcellcnt+relaygen2}=[i,carnum(i)-2*urvrelaysel2+relaygen2];
                end
                urvcellcnt=urvcellcnt+urvrelaysel2;
            end

%             k1=0;
%             count = zeros(10,1);
%             for i=1:4
%                 for j = 1:carnum(i)
%                     if d12_R(i,j)==0
%                         flag = 0;
%                         while flag==0
%                             while bucket_cnt(mod(car_idnum(i,j)+k1,10)+1)==0 %如果对应的桶为空，即没有可以作为relay的车，看下一个桶
%                                 k1 = k1+1;
%                             end
%                             relaybucket=mod(car_idnum(i,j)+k1,10)+1;
%                             count(relaybucket) = mod(count(relaybucket),bucket_cnt(relaybucket))+1;%桶中用来relay的对应车辆，从第一辆开始，桶内循环
%                             tmp=car_bucket{relaybucket,count(relaybucket)}(3);%相应桶的相应车第三个元素，记录的是relay的数目
%                             if tmp<1
%                                 flag = 1;
%                                 car_bucket{relaybucket,count(relaybucket)}(3)=tmp+1;%更新relay信息
%                                 car_bucket{relaybucket,count(relaybucket)}(tmp*2+4)=i;
%                                 car_bucket{relaybucket,count(relaybucket)}(tmp*2+5)=j;
%                                 CRV{urvcellcnt}=car_bucket{relaybucket,count(relaybucket)}(1:2);
%                                 URV{urvcellcnt}=[i,j];
%                                 urvcellcnt=urvcellcnt+1;
%                             else
%                                 k1 = k1+1;%若该车已经有relay了，换下一个桶
%                             end
%                             %count(relaybucket) =
%                             %mod(count(relaybucket),bucket_cnt(relaybucket))+1;换同一个桶中下一辆车
%                             % 还没写完，可以考虑先把同一个桶中的车先找完再找下一个桶，不用了，上面已经考虑到了
%                         end
%                     end
%                 end
%             end

            %%
            %A1=[1,0;0,1;0,0];
            %A2=[1,0;0,0];
            %用cell来构建URV及对应的CRV的index坐标
            % CRV={[2,4],[4,2],[2,2],[1,4],[3,4],[1,2],[2,3]};
            % URV={[1,6],[2,1],[4,1],[3,6],[1,1],[3,1],[4,6]};
            %考虑在找relay的时候建立cell数组
            k=3;
            for i = 1:urvnum
                d23_V(URV{i}(1),URV{i}(2))=sqrt((car_location(URV{i}(1),URV{i}(2))-car_location(CRV{i}(1),CRV{i}(2))^2+(verd(URV{i}(1))-verd(CRV{i}(1)))^2));
            end
%             xhat=zeros(7,3);%7是relay的数量
            matrixd=max(crvnum,urvnum);
            A1=zeros(crvnum,3*matrixd,3); %CRV约束的系数矩阵
            for i=1:crvnum
                A1(i,3*i-2:3*i,:)=eye(3);
            end
            A1T=permute(A1,[2,3,1]);
            A2=zeros(urvnum,3*matrixd,2);%URV约束的系数矩阵
            for i=1:urvnum
                A2(i,3*i-2,1)=1;
            end
            A2T=permute(A2,[2,3,1]);
            C=zeros(3*matrixd,1);%算法中的约束对应的常数矩阵C
            for i=1:crvnum
                C(i*3-2:3*i)=[B2_0;P2_0;P2_1];
            end
%             for gasel=1:7
%                 if rdir1con==rdir2con && gasel <=4
%                     continue;
%                 end
gasel=4;
                disc=zeros(4,35,500);
                x=zeros(3,500,4,35);
                x1=zeros(2,500,4,35);
            for i=1:4
                for j=1:carnum(i)
                    if d12_R(i,j)>0
                        BP1=sdpvar(3,1);
                        F0_1 = [0<=BP1(1)<=B2_0,0<=BP1(2)<=P2_0,0<=BP1(3)<=P2_1];
                        obj0_1=-r(i,j)*cap(BP1(1),BP1(2),d12_R(i,j),ga(gasel),N0)+pP*(BP1(2)+BP1(3))+pB*BP1(1);
                        options = sdpsettings('verbose',0,'cachesolvers',1,'solver','fmincon','fmincon.MaxFunEvals',5000,'fmincon.TolCon',...
                            1e-20,'fmincon.TolX',1e-20,'fmincon.TolFun',1e-20,'fmincon.TolConSQP',1e-20,...
                            'fmincon.TolPCG',1e-20,'fmincon.TolProjCG',1e-20,'fmincon.TolProjCGAbs',1e-20);
                        sol1=optimize(F0_1,obj0_1,options);
                        optobj1=value(obj0_1);
                        optx1=value(BP1);
                        %xhat(i)=optx1;
                        x(:,2,i,j)=optx1;
                    end
                end
            end
            crvdisflag=1;
            urvdisflag=1;
            fx=0;
            for i=1:4
                for j=1:carnum(i)
                    if d12_R(i,j)>0
                        if cap(x(1,2,i,j),x(2,2,i,j),d12_R(i,j),ga(gasel),N0)>=c0
                            fx=fx+1;
                            if crvdisflag ==1
                                disfxwhencrv=[num2str(fx),' crviscnted '];
                                disp(disfxwhencrv);
                                crvdisflag=0;
                            end
                        end
                    end
                end
            end
            fxbefore(rsel,gasel)=fx;
                % x2=zeros(2,500,t);
                %c1=cap(optx1(1),optx1(2),d12_R,ga,N0);
                c2=0;
                lamda=zeros(3*matrixd,500);
                theta=zeros(sum(carnum)*3,500);
                theta(:,3)=ones(sum(carnum)*3,1);
                c_urv=zeros(4,35);
                tonorm=zeros(3*matrixd,1);
                tonormlast=zeros(3*matrixd,1);
                fx1=zeros(500,1);
                fx1(1)=crvnum;
                fx1(2)=crvnum;
                tmpx=zeros(3,1);
                tmpx1=zeros(2,1);
                while k<=10 || abs((-fx1(k-1)-lamda(k-1)*tonorm+rho/2*norm(tonorm)^2)-(-fx1(k-2)-lamda(k-2)*tonormlast+rho/2*norm(tonormlast)^2))>eps%L(x(:,k-1,t),x(:,k-1,t),lamda(:,k-1,t),C,rho,A1,A2,c1,c2)-...
                        %L(x(:,k-2,t),x(:,k-2,t),lamda(:,k-2,t),C,rho,A1,A2,c1,c2)

                    tonormlast=tonorm;
                    for i=2:t-1
                        crvcnt=1;
                        urvcnt=1;
                        tmpx=x(:,k,4,carnum(4));
                        tmpx1=x1(:,k,4,carnum(4));
                        for eachcari=1:4
                %             disp(eachcari);
                            for eachcarj = 1:carnum(eachcari)
                                currentcar=eachcarj;
                                for curcarcnt=1:eachcari-1
                                    currentcar=currentcar+carnum(curcarcnt);
                                end
                                tonorm=-C;
                                urv=1;crv=1;
                                for eachcari1=1:4
                                    for eachcarj1=1:carnum(eachcari1)
                                        if d12_R(eachcari1,eachcarj1)>0
                                            tonorm=tonorm+A1T(:,:,crv)*x(:,k,eachcari1,eachcarj1);
                                            crv=crv+1;
                                        else
                                            tonorm=tonorm+A2T(:,:,urv)*x1(:,k,eachcari1,eachcarj1);
                                            urv=urv+1;
                                        end
                                    end
                                end
                %                 disp(eachcarj);
                                if d12_R(eachcari,eachcarj)>0
                                    %CRV
                                    BP=sdpvar(3,1);
                %                     disp(100);
                                    F1 = [0<=BP(1)<=B2_0,0<=BP(2)<=P2_0,0<=BP(3)<=P2_1];
                            %         obj1=-r2*cap(BP(1),BP(2),d12_R,ga,N0)+pP*(BP(2)+BP(3))+pB*BP(1)-theta(1,k)*BP(1)-theta(2,k)*BP(2)-theta(3,k)*BP(3)-lamda(:,k,i)'*A1'*BP+rho/2*norm(A1'*BP+A2'*x2(:,k,i)-C)^2-M./(1+(exp(-cap(BP(1),BP(2),d12_R,ga,N0))));
                                    obj1=-r(eachcari,eachcarj)*cap(BP(1),BP(2),d12_R(eachcari,eachcarj),ga(gasel),N0)+pP*(BP(2)+BP(3))+pB*BP(1)-theta((currentcar-1)*3+1,k)*BP(1)- theta((currentcar-1)*3+2,k)*BP(2)-theta((currentcar-1)*3+3,k)*BP(3)-lamda(:,k)'*A1T(:,:,crvcnt)*BP+rho/2*norm(A1T(:,:,crvcnt)*BP-A1T(:,:,crvcnt)*x(:,k,eachcari,eachcarj)+tonorm)^2-M*double(cap(BP(1),BP(2),d12_R(eachcari,eachcarj),ga(gasel),N0)>=c0);
                                    options = sdpsettings('verbose',0,'cachesolvers',1,'solver','fmincon','fmincon.MaxFunEvals',7000,'fmincon.TolCon',...
                                        1e-20,'fmincon.TolX',1e-20,'fmincon.TolFun',1e-20,'fmincon.TolConSQP',1e-20,...
                                        'fmincon.TolPCG',1e-20,'fmincon.TolProjCG',1e-20,'fmincon.TolProjCGAbs',1e-20);
                                    sol1=optimize(F1,obj1,options);
                %                     disp(101);
                                    x(:,k,eachcari,eachcarj)=value(BP);
                %                     disp(102);
                                    crvcnt=crvcnt+1;
                                    disc(eachcari,eachcarj,k)=cap(x(1,k,eachcari,eachcarj),x(2,k,eachcari,eachcarj),d12_R(eachcari,eachcarj),ga(gasel),N0);
                %                     disp(crvcnt);
                                else
                                    %URV
                                    BP2=sdpvar(2,1);
                                    relayi=CRV{urvcnt}(1);
                                    relayj=CRV{urvcnt}(2);
%                                     for tmpi=1:10
%                                         if (isempty(car_bucket{tmpi,1})) %判断tmpi号桶中是否有车
%                                             continue;
%                                         else
%                                             for tmpj=1:30 %3应该是可以调大一点的
%                                                 if eachcari==car_bucket{tmpi,tmpj}(4) && eachcarj==car_bucket{tmpi,tmpj}(5)
%                                                     relayi=car_bucket{tmpi,tmpj}(1);
%                                                     relayj=car_bucket{tmpi,tmpj}(2);
%                                                     break;
%                                                 end
%                                                 if(isempty(car_bucket{tmpi,tmpj+1})==1) %如果找不到有relay的车，那么跳出这层循环
%                                                     break;
%                                                 end
%                                             end
%                                             if relayi>0
%                                                 break;
%                                             end
%                                         end
%                                     end
                                    F2=[0<=BP2(1)<=B2_0,0<=BP2(2)<=P3_0];
                            %         obj2=-r3*0.5*min(cap(BP2(1),BP2(2),d12_R,ga,N0),cap(BP2(1),x1(3,k,i+1),d23_V,ga,N0))+pP*...
                            %             BP2(2)+pB*BP2(1)-M*(min(cap(BP2(1),BP2(2),d13,k,i+1),d23_V,ga,N0))...
                            %             -c0)-lamda(:,k,i)'*A2'*BP2+rho/2*norm(A2'*BP2+A1'*x1(:,k,i+1)-C)^2;
                                    if relayi<eachcari ||(relayi==eachcari && relayj<eachcarj)
                                        obj2=-r(eachcari,eachcarj)*0.5*min(cap(BP2(1),BP2(2),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(BP2(1),x(3,k,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0))+pP*...
                                            BP2(2)+pB*BP2(1)-M*double(min(cap(BP2(1),BP2(2),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(BP2(1),x(3,k,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0))...
                                            >=c0)-lamda(:,k)'*A2T(:,:,urvcnt)*BP2+rho/2*norm(A2T(:,:,urvcnt)*BP2-A2T(:,:,urvcnt)*x1(:,k,eachcari,eachcarj)+tonorm)^2;
                                        options = sdpsettings('verbose',0,'cachesolvers',1,'solver','fmincon','fmincon.MaxFunEvals',7000,'fmincon.TolCon',...
                                            1e-20,'fmincon.TolX',1e-20,'fmincon.TolFun',1e-20,'fmincon.TolConSQP',1e-20,...
                                            'fmincon.TolPCG',1e-20,'fmincon.TolProjCG',1e-20,'fmincon.TolProjCGAbs',1e-20);
                                        sol22=optimize(F2,obj2,options);
                                        x1(:,k,eachcari,eachcarj)=value(BP2);
                                        c_urv(eachcari,eachcarj)=0.5*min(cap(x1(1,k,eachcari,eachcarj),x1(2,k,eachcari,eachcarj),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(x1(1,k,eachcari,eachcarj),x(3,k,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0));
                                    else
                                        obj2=-r(eachcari,eachcarj)*0.5*min(cap(BP2(1),BP2(2),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(BP2(1),x(3,k-1,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0))+pP*...
                                            BP2(2)+pB*BP2(1)-M*double(min(cap(BP2(1),BP2(2),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(BP2(1),x(3,k-1,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0))...
                                            >=c0)-lamda(:,k)'*A2T(:,:,urvcnt)*BP2+rho/2*norm(A2T(:,:,urvcnt)*BP2-A2T(:,:,urvcnt)*x1(:,k,eachcari,eachcarj)+tonorm)^2;
                                        options = sdpsettings('verbose',0,'cachesolvers',1,'solver','fmincon','fmincon.MaxFunEvals',7000,'fmincon.TolCon',...
                                            1e-20,'fmincon.TolX',1e-20,'fmincon.TolFun',1e-20,'fmincon.TolConSQP',1e-20,...
                                            'fmincon.TolPCG',1e-20,'fmincon.TolProjCG',1e-20,'fmincon.TolProjCGAbs',1e-20);
                                        sol2=optimize(F2,obj2,options);
                                        x1(:,k,eachcari,eachcarj)=value(BP2);
                                        c_urv(eachcari,eachcarj)=0.5*min(cap(x1(1,k,eachcari,eachcarj),x1(2,k,eachcari,eachcarj),sqrt(car_location(eachcari,eachcarj)^2 + verd(eachcari)^2),ga(gasel),N0),cap(x1(1,k,eachcari,eachcarj),x(3,k-1,relayi,relayj),d23_V(eachcari,eachcarj),ga(gasel),N0));
                                    end
                                    urvcnt=urvcnt+1;
                %                     disp(urvcnt);
                                end
                %                 disp(eachcarj);
                            end
                        end
                        tonorm=tonorm-A2T(:,:,urvnum)*tmpx1+A2T(:,:,urvnum)*x1(:,k,4,carnum(4));
                        lamda(:,k)=lamda(:,k)-rho*(tonorm);
                    end
                %     c1=cap(x1(1,k,t),x1(2,k,t),d12_R,ga,N0);
                %     disc(1,k)=c1;
                %     c2=min(cap(x2(1,k,t),x2(2,k,t),d12_R,ga,N0),cap(x2(1,k,t),x1(3,k,t),d23_V,ga,N0));
                %     disc(2,k)=c2;
                    flag=0;
                    rescrv=1;
                %     disp(c_urv);
                    for i=1:4
                        for j=1:carnum(i)
                            if d12_R(i,j)>0
                                theta(3*rescrv-3+1,k+1)=(pB-r(i,j)*(log2(1+x(2,k,i,j).*d12_R(i,j)^(-ga(gasel))./(N0*x(1,k,i,j)))-1/(log(2)*(1+(N0*x(1,k,i,j))/x(2,k,i,j).*d12_R(i,j)^(-ga(gasel))))))*alpha+(1-alpha)*theta(3*rescrv-3+1,k);
                                theta(3*rescrv-3+2,k+1)=(pP-r(i,j)/log(2)*(d12_R(i,j)^(-ga(gasel))./N0+x(2,k,i,j)/x(1,k,i,j)))*alpha+(1-alpha)*theta(3*rescrv-3+2,k);
                                theta(3*rescrv,k+1)=theta(3*rescrv,k)*(1-alpha)+pP*alpha;%-(pP-r(i,j)/log(2)*(d12_R(i,j)^(-ga(gasel))./N0+x(2,k,i,j)/x(1,k,i,j)))*alpha;
                                rescrv=rescrv+1;
                            end
                        end
                    end
                    instrucsion='count begin ';
                    disp(instrucsion);
                    crvdisflag1=1;
                urvdisflag1=1;
                for i=1:4
                    for j=1:carnum(i)
                        if d12_R(i,j)>0
                            if cap(x(1,k,i,j),x(2,k,i,j),d12_R(i,j),ga(gasel),N0)>=c0
                                fx1(k)=fx1(k)+1;
                                if crvdisflag1 ==1
                                    disfxwhencrv=[num2str(fx1(k)),' crviscnted ',num2str(k)];
                                    disp(disfxwhencrv);
                                    crvdisflag1=0;
                                end
                            end
                        else
                            if c_urv(i,j)>=c0
                                fx1(k)=fx1(k)+1;
                                if urvdisflag1==1
                                    disfxwhenurv=[num2str(fx1(k)),' urviscnted ',num2str(k)];
                                    disp(disfxwhenurv);
                                    urvdisflag1=0;
                                end
                            end
                        end
                    end
                end

                    for j = 1:urvnum
                        if abs(disc(CRV{j}(1),CRV{j}(2),k)-disc(CRV{j}(1),CRV{j}(2),k-1))<eps2/rsel
                            flag = flag+1;
                        end
                    end
                    if flag==urvnum &&k>=7
                        break;
                    end
                    k=k+1;
                    disp(k);
                end
                fxstar(:,rsel,gasel)=fx1;
%            end
%     end
end