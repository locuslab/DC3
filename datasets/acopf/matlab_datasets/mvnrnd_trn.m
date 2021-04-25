function smpOut = mvnrnd_trn(lb,ub,mu,sigMat,NUM_smp)
%Sample truncated multivariate normal distribution
%INPUTS
%   lb:        lower bound vector [1,d] 
%   ub:        upper bound vector [1,d]
%   mu:        means vectors [1,d]
%   sigMat:    covariance matrix [d,d]
%   NUM_smp:   number of samples
%OUTPUTS
%   smpOut:    matrix of all samles [NUM_smp,d]
%
%Example:
%smpOut = mvnrnd_trn(zeros(1,3),ones(1,3)+1,0.3+rand(1,3)/2,eye(3),1000);
%hist(smpOut(:,1))
%
%Y.Kamer
%2018.10.24
%Zurich
smpOut  = nan(NUM_smp,numel(lb));
NUM_try = NUM_smp;  % trial size
NUM_cur	= 0;        % current accepted number
    while NUM_cur<NUM_smp
        tmpSmp  = mvnrnd(mu,sigMat,NUM_try);                    % sample MVN
        indAcc  = all(tmpSmp > repmat(lb,[NUM_try 1]),2) & ...  % check if within LB and UB
                  all(tmpSmp < repmat(ub,[NUM_try 1]),2);
        numAcc  = sum(indAcc); 
        if(numAcc)              % record accepted samples
            iS  = NUM_cur+1;
            iE  = iS + numAcc - 1;
            if(iE>NUM_smp)
                iE      = NUM_smp;
                indAcc  = find(indAcc);
                indAcc  = indAcc(1:(iE-iS+1));
            end
            smpOut(iS:iE,:) = tmpSmp(indAcc,:);
        end
        NUM_cur = NUM_cur + numAcc;                         % increment counter 
        NUM_try = round((NUM_smp-numAcc) * NUM_try/numAcc); % estimate next trial size
        NUM_try = min(NUM_try,10*NUM_smp);                  % limit max trial size
    end
end