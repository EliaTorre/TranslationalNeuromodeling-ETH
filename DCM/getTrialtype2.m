function trialtype = getTrialtype2(tones)
%GETTRIALTYPE classifies trials as ["undefined", "standard", "deviant"]
%
%   tones: binary vector of tones as in stimulus_sequence.mat
%
% returns:
%   stddev: categorical vector with categories ["undefined", "standard",
%   "deviant"] of same size as tones

arguments
    tones (1,:) double
end

stable_phases = [1:300, 351:400, 901:1200, 1251:1350];

trialtype = strings(size(tones))+"undefined";

rep = 1;
for i=2:length(tones)
    if tones(i)==tones(i-1)
        if rep==5
            trialtype(i) = "standard5";
        elseif rep>=12
            trialtype(i) = "standard12+";
        end
        rep = rep+1; 
    else
        if rep >=3 && rep<=5 
            trialtype(i) = "deviant3-5";
        elseif rep >= 6
            trialtype(i) = "deviant6+";
        end
        rep = 1;
    end
end

trialtype = categorical(trialtype);


