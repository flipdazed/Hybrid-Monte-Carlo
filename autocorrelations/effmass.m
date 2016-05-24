function [f] = effmass(G,t1,t2)
%
% call:
% [f] = effmass(G,t1,t2)
%
% effective mass from correlation G (vector) by forming
% f=-log(G(t2)/G(t1))/(t2-t1)
% legitimacy of arguments is NOT checked
% test-function for routine UWerr.m
%
%----------------------------------------------------------------------------
%  Ulli Wolff,   June 2003, Version 1
%----------------------------------------------------------------------------

f=-log(G(t2)/G(t1))/(t2-t1);
