function RegistersinCellarray = convert_Presilicon2MatlabRegisters(fileName)
    PreSiliconRegisters=importfile_presil(fileName);
    numberofRegisters=length(PreSiliconRegisters(:,1));
    RegistersinCellarray=cell(numberofRegisters,2);
    for i=1:numberofRegisters
            RegistersinCellarray{i,1}=PreSiliconRegisters(i,1);
            RegistersinCellarray{i,2}=hex2dec(PreSiliconRegisters(i,4));
    end
end

function CWModewithDACControlTX224125GHz = importfile_presil(filename, dataLines)
%IMPORTFILE Import data from a text file
%  CWMODEWITHDACCONTROLTX224125GHZ = IMPORTFILE(FILENAME) reads data
%  from text file FILENAME for the default selection.  Returns the data
%  as a string array.
%
%  CWMODEWITHDACCONTROLTX224125GHZ = IMPORTFILE(FILE, DATALINES) reads
%  data for the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  CWModewithDACControlTX224125GHz = importfile("C:\Users\Savas\Documents\BGT24ATR22\Matlab_220301_CWModes\atr22\examples\CW_Mode_with_DAC_Control_TX2_24_125GHz.txt", [1, Inf]);
%
%  See also READMATRIX.
%
% Auto-generated by MATLAB on 01-Mar-2022 16:09:03

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["RegisterName", "RegAddress", "ResetValues", "RegValues"];
opts.VariableTypes = ["string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Specify variable properties
opts = setvaropts(opts, ["RegisterName", "RegAddress", "ResetValues", "RegValues"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["RegisterName", "RegAddress", "ResetValues", "RegValues"], "EmptyFieldRule", "auto");

% Import the data
CWModewithDACControlTX224125GHz = readmatrix(filename, opts);

end