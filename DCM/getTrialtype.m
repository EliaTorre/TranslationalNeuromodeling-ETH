function stddev = getTrialtype(tones, repetitions_before_standard, volatility)
%GETTRIALTYPE classifies trials as ["undefined", "standard", "deviant"]
%
%   tones: binary vector of tones as in stimulus_sequence.mat
%
% returns:
%   stddev: categorical vector with categories ["undefined", "standard",
%   "deviant"] of same size as tones

arguments
    tones (1,:) double
    repetitions_before_standard = 5
    volatility = false;
end

stable_phases = [1:300, 351:400, 901:1200, 1251:1350];

stddev = strings(size(tones))+"undefined";

rep = 1;
for i=2:length(tones)
    if tones(i)==tones(i-1)
        if rep==repetitions_before_standard
            stddev(i) = "standard";

            if volatility
                if any(stable_phases==i)
                    stddev(i) = stddev(i) + ", stable";
                else
                    stddev(i) = stddev(i) + ", volatile";
                end
            end
        end
        rep = rep+1; 
    else
        if rep>=repetitions_before_standard
            stddev(i) = "deviant";

            if volatility
                if any(stable_phases==i)
                    stddev(i) = stddev(i) + ", stable";
                else
                    stddev(i) = stddev(i) + ", volatile";
                end
            end
        end
        rep = 1;
    end
end

stddev = categorical(stddev);


