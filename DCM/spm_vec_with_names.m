function [vX, names] = spm_vec_with_names(X,varargin)
% Vectorise a numeric, cell or structure array - a compiled routine
% FORMAT [vX] = spm_vec(X)
% X  - numeric, cell or stucture array[s]
% vX - vec(X)
%
% See spm_unvec
%__________________________________________________________________________
%
% e.g.:
% spm_vec({eye(2) 3}) = [1 0 0 1 3]'
%__________________________________________________________________________
% Copyright (C) 2005-2013 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_vec.m 6110 2014-07-21 09:36:13Z karl $


%error('spm_vec.c not compiled - see Makefile')

% initialise X and vX
%--------------------------------------------------------------------------
if nargin > 1
    X = [{X},varargin];
end


% vectorise numerical arrays
%--------------------------------------------------------------------------
if isnumeric(X)
    vX = X(:);
    if isscalar(X)
        names = {''};
    elseif isvector(X)
        names = cell(length(X),1);
        for m=1:length(X)
            names{m} = sprintf('(%i)', m);
        end
    else
        names = cell(size(X));
        for n=1:size(X,2)
            for m=1:size(X,1)
                names{m,n} = sprintf('(%i,%i)', m, n);
            end
        end
        names = names(:);
    end

% vectorise structure into cell arrays
%--------------------------------------------------------------------------
elseif isstruct(X)
    vX = [];
    f   = fieldnames(X);
    names = {};
    X    = X(:);
    for i = 1:numel(f)
        [xx, na] = spm_vec_with_names(X.(f{i}));
        vX = cat(1,vX, xx);
        names = [names; strcat(f{i},na)];
    end

% vectorise cells into numerical arrays
%--------------------------------------------------------------------------
elseif iscell(X)
    vX   = [];
    names = {};
    for i = 1:numel(X)
        [xx, na] = spm_vec_with_names(X{i});
        vX = cat(1,vX,xx);
        names = [names; strcat(sprintf('{%i}',i), na)];
    end
else
    vX = [];
end
