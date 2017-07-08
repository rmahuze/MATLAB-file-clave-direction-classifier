% PARZEN PROBABILISTIC NEURAL NETWORK (PPNN) 
% MACHINE LEARNING II: EE 473
% DATE: 5/7/2016
% RICHARD MAHUZE

% CLASSIFICATION DATASET FROM:
% http://archive.ics.uci.edu/ml/datasets/Firm-Teacher_Clave-Direction_Classification

% 10800 Instances
% 20 Bit/Attributes: 16 Bit Data, 4 bit one-hot encoding Output/Class

% The program will have 4 layers: 
% Input, Pattern, Summation, and Output

% The input layer's neurons are as many as the variables of the system, and
% all of them are connected to the neurons in the next layer.

% The second layer is the pattern layer, which has as many units as the
% number of samples in the training set. Each unit model a Gaussian
% function centered on a training sample. The output for each unit is
% connected to the next layer, in which its own class.

% The third layer has as many neurons as the number of classes in the data. 
% This layer performs probability calculation for input X to each class.

% The output layer will decide which class the input belongs to, based on
% the highest value vote

% Example: 
% Inputs = [
%           0 1 1 1 0.5 0 0.6 1 0.8 0 1 1 0.1 0 1 0.2
%          ];

% The outcomes are the following:
%     
% Class 0 score is 0.115611 
% Class 1 score is 0.282992 
% Class 2 score is 0.070122 
% Class 3 score is 0.115611
% 
%     The input belongs to Class 1, with score 0.28.
% 
%     Elapsed time is ___ seconds.

% The program uses a brute force approach by implementing for loops and
% simple matrix manipulation. Further debugging is needed to apply
% vectorization

% Issues:
% Not being able to use all 10800 instances of data. Further debugging
% needed to produce a confusion matrix to see the program's sensitivity,
% specitivity, reliability, precision and accuracy.



tic;

% fid = fopen('ClaveVectors_Firm-Teacher_Model.txt', 'r');
data = load('ClaveVectors_Firm-Teacher_Model.txt');

data_size = size(data);
NumberOfVectors = data_size(1,1);
NumberOfBits = data_size(1,2);

Class0_Output = [0 0 0 1];      % Incoherent
Class1_Output = [0 0 1 0];      % Forward Clave
Class2_Output = [0 1 0 0];      % Reverse Clave
Class3_Output = [1 0 0 0];      % Neutral

% NumberOfClass0 = 0;
% NumberOfClass1 = 0;
% NumberOfClass2 = 0;
% NumberOfClass3 = 0;
Class0 = zeros(NumberOfVectors,16);
Class1 = zeros(NumberOfVectors,16);
Class2 = zeros(NumberOfVectors,16);
Class3 = zeros(NumberOfVectors,16);


% INPUT 
Inputs = [
          
             0 1 0 0 0 1 0 0 0 1 0 1 0 0 0 0    
             
         ];

tf0 = 0;
tf1 = 0;
tf2 = 0;
tf3 = 0;



% DATA GROUPING TO EACH CLASS 0,1,2,3 BASED ON OUTPUT: FIRST LAYER
for i = 1:NumberOfVectors
    

    
    tf0 = isequal(data(i,17:20), Class0_Output(1,:));
    tf1 = isequal(data(i,17:20), Class1_Output(1,:));
    tf2 = isequal(data(i,17:20), Class2_Output(1,:));
    tf3 = isequal(data(i,17:20), Class3_Output(1,:));

        
        if tf0 == 1     
     %     NumberOfClass0 = NumberOfClass0 + 1;      %Class 0
     %     for j = 1:NumberOfClass0
              
             
             Class0(i,1:16) = data(i,1:16); 
             
             
     %     end
        end
        
        if tf1 == 1
    %      NumberOfClass1 = NumberOfClass1 + 1;      %Class 1
    %      for k = 1:NumberOfClass1  
              
             Class1(i,1:16) = data(i,1:16); 
    %      end
        end
        
        if tf2 == 1
    %      NumberOfClass2 = NumberOfClass2 + 1;      %Class 2
    %     for l = 1:NumberOfClass2  
             
            Class2(i,1:16) = data(i,1:16); 
    %      end
        end
        
        if tf3 == 1
    %      NumberOfClass3 = NumberOfClass3 + 1;      %Class 3
    %      for m = 1:NumberOfClass3  
            Class3(i,1:16) = data(i,1:16); 
    %      end
        end
        
        
end


Class0( ~any(Class0,2), : ) = []; % delete zero rows from Class0
Class1( ~any(Class1,2), : ) = []; % delete zero rows from Class1
Class2( ~any(Class2,2), : ) = []; % delete zero rows from Class2
Class3( ~any(Class3,2), : ) = []; % delete zero rows from Class3



% Showing that not all 10800 data being used
% DataIndex = size(Class0,1)+size(Class1,1)+size(Class2,1)+size(Class3,1);
% if ~isequal(DataIndex,NumberOfVectors)
%     sprintf 'NOT 10800 due to data discrepancy'
% end



% if I want to Randomized them

Class0(randperm(size(Class0,1)),:) = Class0;
Class1(randperm(size(Class1,1)),:) = Class1;
Class2(randperm(size(Class2,1)),:) = Class2;
Class3(randperm(size(Class3,1)),:) = Class3;

% randomize each element
% Class0(randperm(numel(Class0))) = Class0;
% Class1(randperm(numel(Class1))) = Class1;
% Class2(randperm(numel(Class2))) = Class2;
% Class3(randperm(numel(Class3))) = Class3;

% randomize just the rows
%  Class0_FIVE(randperm(size(Class0_FIVE,1)),:) = Class0_FIVE;
%  Class1_FIVE(randperm(size(Class1_FIVE,1)),:) = Class1_FIVE;
%  Class2_FIVE(randperm(size(Class2_FIVE,1)),:) = Class2_FIVE;
%  Class3_FIVE(randperm(size(Class3_FIVE,1)),:) = Class3_FIVE;




% PICK FIVE VECTORS FROM EACH CLASS: SOME, NOT ALL
Class0_Index = 100;
Class1_Index = 100;
Class2_Index = 100;
Class3_Index = 100;

Class0_FIVE = Class0(1:Class0_Index,:);
Class1_FIVE = Class1(1:Class1_Index,:);
Class2_FIVE = Class2(1:Class2_Index,:);
Class3_FIVE = Class3(1:Class3_Index,:);

% randomize just the rows
% Class0_FIVE(randperm(size(Class0_FIVE,1)),:) = Class0_FIVE;
% Class1_FIVE(randperm(size(Class1_FIVE,1)),:) = Class1_FIVE;
% Class2_FIVE(randperm(size(Class2_FIVE,1)),:) = Class2_FIVE;
% Class3_FIVE(randperm(size(Class3_FIVE,1)),:) = Class3_FIVE;


% VARIABLES DECLARATION FOR CALCULATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


D_M_E_SQ_0 = zeros(Class0_Index,16);
DV_BY_2_0 = zeros(Class0_Index,16);
SU_EL_0 = zeros(Class0_Index,1);
SU_EXP_0 = 0;

D_M_E_SQ_1 = zeros(Class1_Index,16);
DV_BY_2_1 = zeros(Class1_Index,16);
SU_EL_1 = zeros(Class1_Index,1);
SU_EXP_1 = 0;

D_M_E_SQ_2 = zeros(Class2_Index,16);
DV_BY_2_2 = zeros(Class2_Index,16);
SU_EL_2 = zeros(Class2_Index,1);
SU_EXP_2 = 0;

D_M_E_SQ_3 = zeros(Class3_Index,16);
DV_BY_2_3 = zeros(Class3_Index,16);
SU_EL_3 = zeros(Class3_Index,1);
SU_EXP_3 = 0;


Class0_Score = 0;
Class1_Score = 0;
Class2_Score = 0;
Class3_Score = 0;
Classes_Score = zeros(1,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%


% PATTERN LAYER: SECOND LAYER
for o=1:Class0_Index      % o and p
    
    
    for p=1:16
        
    D_M_E_SQ_0(o,p) = (Class0_FIVE(o,p) - Inputs(p)).^2;
    DV_BY_2_0(o,p) = D_M_E_SQ_0(o,p)/2;
    
        
    end
    
    SU_EL_0(o) = sum(DV_BY_2_0(o,1:16));
    SU_EXP_0 = SU_EXP_0 + exp(-(SU_EL_0(o)));
    
     
end



for q=1:Class1_Index       % q and r
    
    
    for r=1:16
        
    D_M_E_SQ_1(q,r) = (Class1_FIVE(q,r) - Inputs(r)).^2;
    DV_BY_2_1(q,r) = D_M_E_SQ_1(q,r)/2;
    
        
    end
    
    SU_EL_1(q) = sum(DV_BY_2_1(q,1:16));
    SU_EXP_1 = SU_EXP_1 + exp(-(SU_EL_1(q)));
    
     
end



for s=1:Class2_Index       % s and t
    
    
    for t=1:16
        
    D_M_E_SQ_2(s,t) = (Class2_FIVE(s,t) - Inputs(t)).^2;
    DV_BY_2_2(s,t) = D_M_E_SQ_2(s,t)/2;
    
        
    end
    
    SU_EL_2(s) = sum(DV_BY_2_2(s,1:16));
    SU_EXP_2 = SU_EXP_2 + exp(-(SU_EL_2(s)));
    
     
end



for u=1:Class3_Index       % u and v
    
    
    for v=1:16
        
    D_M_E_SQ_3(u,v) = (Class3_FIVE(u,v) - Inputs(v)).^2;
    DV_BY_2_3(u,v) = D_M_E_SQ_3(u,v)/2;
    
        
    end
    
    SU_EL_3(u) = sum(DV_BY_2_3(u,1:16));
    SU_EXP_3 = SU_EXP_3 + exp(-(SU_EL_3(u)));
    
     
end



% CALCULATE PROBABILITY SCORE: THIRD LAYER
Class0_Score = SU_EXP_0/Class0_Index;
Class1_Score = SU_EXP_1/Class1_Index;
Class2_Score = SU_EXP_2/Class2_Index;
Class3_Score = SU_EXP_3/Class3_Index;


Classes_Score(1,1) = Class0_Score;
Classes_Score(1,2) = Class1_Score;
Classes_Score(1,3) = Class2_Score;
Classes_Score(1,4) = Class3_Score;


% FIND ARGMAX FROM THE CLASSES_SCORE OUTPUT: FOURTH LAYER

[argmax] = max(Classes_Score);

% Display score from each class
for z = 1:4
    
 fprintf('Class %i score is %f \n', z-1, Classes_Score(z))

end

for w = 1:4
    

    if isequal(argmax,Classes_Score(1,w))
        
    formatSpec = 'The input belongs to Class %d, with score %f.';
    str = sprintf(formatSpec,w-1,Classes_Score(1,w))
    
    end
        
        
end



toc;


