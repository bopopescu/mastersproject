function [ FracturePerMeter ] = importFracDensity

filename = 'FracturePerMeter.txt';
delimiter = '\t';

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: categorical (%C)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%C%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
FracturePerMeter = table(dataArray{1:end-1}, 'VariableNames', {'Depthm','FracCount','Borehole'});


%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp;


end

