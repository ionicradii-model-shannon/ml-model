clear
clc

%% Reading the table

data=readtable('ShData.xlsx');
data=table2array(data);

%% Control

kf=3; % kfold selection
RMF=2; % feature selection
XFlag=21; % regression model selection

%% optimization options
hyperopts = struct('MaxObjectiveEvaluations',50,'AcquisitionFunctionName','probability-of-improvement','Repartition',true);
opts.TolX=1e-12;
opts.MaxFunEvals=1000;

Y=data(:,end);
XFull=data(:,1:end-1);
X1Ext=1./XFull(:,[2 5 10]);
X2Ext=exp(XFull(:,[10 11]));
X3Ext=exp(-XFull(:,[10 11]));
X4Ext=log(XFull(:,10));
dataF=[XFull X1Ext X2Ext X3Ext X4Ext Y];

NL=length(Y);
[nr,nc]=size(data);
NF=nc-1;

IndxV=mod(1:NL,kf);
IndxV(IndxV==0)=kf;

%% For k folds
for k=1:kf
    [k kf]
    
    % Initialization
    
    InTest=find(IndxV==k);
    DTest=dataF(InTest,:);
    InTrain=find(IndxV~=k);
    DTrain=dataF(InTrain,:);
    
    %% [1)atomic# 2)r_atomic 3)OS# 4)CN# 5)period# 6)s 7)p 8)d 9)f 10)ion_poten. 11)e_affinity r_ionic_shannon]
    
    switch XFlag
        
        case 1
            pv=1:11;
        case 2
            pv=[1 3:11];
        case 3
            pv=3:11;
        case 4
            pv=3:9;
        case 5
            pv=[1:14];
        case 6
            pv=[1:11 15:16];
        case 7
            pv=[1:11 17:18];
        case 8
            pv=[1:11 19];
        case 9
            pv=1:19;
        case 10
            pv=[1 3:4 6:9 12:14];
        case 11
            pv=[1 3:4 6:9 12:14 15:16];
        case 12
            pv=[1 3:4 6:9 12:14 17:18];
        case 13
            pv=[1 3:4 6:9 12:14 19];
        case 14
            pv=[1 3:4 6:9 15:16];
        case 15
            pv=[1 3:4 6:9 17:18];
        case 16
            pv=[1 3:4 6:9 19];
        case 20
            pv=[3:9 11 17];
        case 21
            pv=[3:9 17];

    end
    
    XTest=DTest(:,pv);
    XTrain=DTrain(:,pv);  
    YTest=DTest(:,end);
    YTrain=DTrain(:,end);  
    
    %% Regression model selection
    
    switch RMF
        case 1
            MdlFinal = fitglm(XTrain,YTrain);
        case 11
            MdlFinal = fitrlinear(XTrain,YTrain,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',hyperopts);
            figure(1)
            figure(2)
            close 1 2  
        case 2
            %HFun=@(X) [ones(size(X)),X,X.^2,X.^3];
            MdlFinal = fitrgp(XTrain,YTrain,'Standardize',true,'BasisFunction','constant','KernelFunction','ardmatern32','Sigma',0.005);%,,'ardsquaredexponential');%'Optimizer','fminunc','Sigma',0.01);
            %MdlFinal = fitrgp(XTrain,YTrain,'Standardize',false,'BasisFunction','linear','KernelFunction','ardmatern32','Sigma',0.1);
      
        case 12
            %MdlFinal = fitrgp(XTrain,YTrain,'Sigma',0.000001,'Standardize',true,'OptimizeHyperparameters',{'BasisFunction','KernelFunction','KernelScale'},'HyperparameterOptimizationOptions',hyperopts);
            MdlFinal = fitrgp(XTrain,YTrain,'Standardize',true,'BasisFunction','pureQuadratic','KernelFunction','ardmatern32','OptimizeHyperparameters',{'Sigma'},'HyperparameterOptimizationOptions',hyperopts);
            figure(1)
            figure(2)
            close 1 2   
        case 3
            MdlFinal = fitrsvm(XTrain,YTrain,'Standardize',false,'KernelFunction','polynomial','PolynomialOrder',2);
        case 13
            MdlFinal = fitrsvm(XTrain,YTrain,'Epsilon',0.01,'Standardize',false,'KernelFunction','polynomial','PolynomialOrder',4,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',hyperopts);
            figure(1)
            figure(2)
            close 1 2     
        case 23
            MdlFinal = fitrsvm(XTrain,YTrain,'KernelFunction','gaussian','Standardize',false,'OptimizeHyperparameters',{'Epsilon','KernelScale'},'HyperparameterOptimizationOptions',hyperopts);
            figure(1)
            figure(2)
            close 1 2    
        case 4
            MdlFinal = ridge(XTrain,YTrain,[0.1 0.5 0.9]);
        case 5

            MdlFinal = fitrkernel(XTrain,YTrain,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',hyperopts);
            figure(1)
            figure(2)
            close 1 2  
    end
    
    % Validation
    
    YTrainFit=predict(MdlFinal,XTrain);
    YTestFit=predict(MdlFinal,XTest);
    ValAcc(k)=sqrt(sum((YTrainFit-YTrain).^2)/length(YTrain))
    TestAcc(k)=sqrt(sum((YTestFit-YTest).^2)/length(YTest))
    
    
end

%% Results
ValBest=min(ValAcc)
ValMean=mean(ValAcc)

TestBest=min(TestAcc)
TestMean=mean(TestAcc)


