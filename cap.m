function [ C ] = cap(B, P,d, gamma, N0)
C=B.*log2(1+P.*d^(-gamma)./(N0*B));


end

