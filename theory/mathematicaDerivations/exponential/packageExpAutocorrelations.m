(* ::Package:: *)

BeginPackage["packageExpAutocorrelations`"];
ghmcLtDerived::usage = "ghmcLtDerived[\[Beta]_, \[Phi]_, \[Theta]_,\[Rho]_, \[Tau]_] is the full derived form of the Laplace Transformed GHMC autocorrelation function"; 
hmcLtDerived::usage = "hmcLtDerived[\[Beta]_, \[Phi]_,\[Rho]_, \[Tau]_] is the full derived form of the Laplace Transformed HMC autocorrelation function"; 
ghmcLt::usage = "ghmc[\[Beta]_, \[Phi]_, \[Theta]_, \[Rho]_, r_] is the Laplace transformed GHMC autocorrelation function"; 
ghmcLt1Acc::usage = "ghmc1Acc[\[Beta]_, \[Phi]_, \[Theta]_, r_] is the Laplace transformed GHMC autocorrelation function with unit acceptance probability"; 
hmcLt::usage = "hmc[\[Beta]_, \[Phi]_, \[Rho]_, r_] is the Laplace transformed HMC autocorrelation function"; 
hmcLt1Acc::usage = "hmc1Acc[\[Beta]_, \[Phi]_, r_] is the Laplace transformed HMC autocorrelation function with unit acceptance probability"; 
ighmc::usage = "ighmc[\[Phi]_, \[Theta]_, \[Rho]_] is the GHMC integrated autocorrelation function (note this is the inverted Laplace Transform)"; 
ighmc1Acc::usage = "ighmc[\[Phi]_, \[Theta]_] is the GHMC integrated autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
ihmc::usage = "ihmc[\[Phi]_, \[Rho]_] is the HMC integrated autocorrelation function (note this is the inverted Laplace Transform)"; 
ihmc1Acc::usage = "ihmc1Acc[\[Phi]_] is the HMC integrated autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
ghmc::usage = "ghmc[t_,\[Phi]_, \[Theta]_, \[Rho]_, r_] is the GHMC autocorrelation function (note this is the inverted Laplace Transform)"; 
ghmc1Acc::usage = "ghmc[t_,\[Phi]_,\[Theta]_,r_] is the GHMC autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
hmc::usage = "hmc[t_,\[Phi]_,\[Rho]_,r_] is the HMC autocorrelation function (note this is the inverted Laplace Transform)"; 
hmc1Acc::usage = "hmc1Acc[t_, \[Phi]_,r_ ] is the HMC autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
ghmcNormalised::usage="ghmcNormalised[t_,\[Phi]_, \[Theta]_, \[Rho]_, r_] is the Normalised GHMC autocorrelation function (note this is the inverted Laplace Transform)";
ghmc1AccNormalised::usage="ghmc1AccNormalised[t_, \[Phi]_, \[Theta]_, r_] is the Normalised GHMC autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)";
hmcNormalised::usage="hmcNormalised[t_,\[Phi]_,\[Rho]_,r_] is the Normalised HMC autocorrelation function (note this is the inverted Laplace Transform)";
hmc1AccNormalised::usage="hmc1AccNormalised[t_, \[Phi]_, r_] is the Normalised HMC autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)";
Begin["`Private`"]; 


Clear[\[Tau],r,\[Phi],\[Theta],\[Rho], tf, steps, dtau, traj, rate, j , k, \[Mu], \[Nu], pacc];
Clear[ F, rawF, Funit, FHMC, FHMCunit];
Clear[iF, iFunit,iFHMC, iFHMCunit,iFHMC, iFHMCunit];
Clear[A, Aunit, AHMC, AHMCunit];
Clear[CF, CFunit, CHMC, CHMCunit];
Clear[Cunit, Cunitn];
Clear[CFn, CFunitn, CHMCn, CHMCunitn];
Clear[tmp0, tmp1, tmp2, hmcTmp2, hmcTmp];
Clear[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9];
Clear[ghmcLtDerived, hmcLtDerived];
Clear[ghmcLt, ghmcLt1Acc, hmcLt,hmcLt1Acc];
Clear[ighmc, ighmc1Acc, ihmc, ihmc1Acc];
Clear[ghmcFullForm, ghmc1AccFullForm, hmcFullForm,hmc1AccFullForm];
Clear[ghmc,ghmc1Acc,hmc,hmc1Acc];
Clear[ghmcNormalised,ghmc1AccNormalised,hmcNormalised,hmc1AccNormalised];


Clear[B, g, gp, gm, gp20, gm00, gp11, gp02];
Clear[gn0, gn1, gn2, gn3, gd0, gd1, gd2, gd3];
Clear[fG, fGHMC];


B[k_] := If[IntegerQ[k/2],\[Beta] \[Tau]+1, (j+k-2 \[Mu] - 2 \[Nu])\[Phi]];
g [j_,k_]:=Sum[Sum[Binomial[j,\[Mu]]  Binomial[k,\[Nu]] ((1/2)^(j+k) (-1)^(\[Nu]+(1/2 k)) B[k])/((\[Beta] \[Tau]+1)^2+\[Phi]^2 (j+k-2\[Mu]-2\[Nu])^2), {\[Nu], 0, k}],{\[Mu], 0, j}];
gp[j_, k_] := \[Rho]  g[ j,k];
gm[j_, k_] := (1-\[Rho])g[j,k];


gp20= gp[2,0];
gm00 = gm[0,0];
gp11 = gp[1,1];
gp02 = gp[0,2];


gn0:=gp20+gm00;
gn1:=-gp20^2-2 gp11^2+gp02 gm00+gp02 gp20+gm00^2;
gn2:=-(gp20-gp02+gm00) (gm00+gp20+gp02);
gn3:=(gm00+gp20+gp02) (gp20^2-2 gp02 gp20+gp02^2+4 gp11^2-gm00^2);
gd0:=1-gm00-gp20;
gd1:=-(gp02 gm00-2 gp11^2+gm00^2+gp20 gp02-gm00+gp20-gp20^2-gp02);
gd2:=-(-gp20^2+gm00-2 gp20 gm00+gp02^2-gm00^2+gp20);
gd3:=2 gp11^2-gp02^3-4 gp11^2 gm00+gp02 gp20^2-gp02 gp20+gp02^2 gp20-gp02 gm00+gp20 gm00^2-gp20^3+gp20^2-gm00^2-4 gp20 gp11^2-gp20^2 gm00-4 gp11^2 gp02+gp02 gm00^2-gp02^2 gm00+gm00^3+2 gp02 gp20 gm00;


ghmcLtDerived[\[Beta]_, \[Phi]_, \[Theta]_,\[Rho]_, \[Tau]_]:=Evaluate[(gn0+gn1 Cos[\[Theta]] + gn2 Cos[\[Theta]]^2+gn3 Cos[\[Theta]]^3)/(gd0+gd1 Cos[\[Theta]]+gd2 Cos[\[Theta]]^2+ gd3 Cos[\[Theta]]^3)];


hmcLtDerived[\[Beta]_, \[Phi]_,\[Rho]_, \[Tau]_] :=Evaluate[(gp20 + gm00)/(1-gp20 - gm00)];


a0[\[Theta]_,\[Rho]_]:=-Cos[\[Theta]]^2+(1-2\[Rho])Cos[\[Theta]]+4;
a1[\[Theta]_,\[Rho]_] :=(-1+2\[Rho])Cos[\[Theta]]^3-3Cos[\[Theta]]^2+(3-6\[Rho])Cos[\[Theta]]+6;
a2[\[Theta]_,\[Rho]_, \[Phi]_] :=(4\[Rho]-2)Cos[\[Theta]]^3+((-2+2\[Rho])(\[Rho]-2)\[Phi]^2+3-6\[Rho])Cos[\[Theta]] + ((4\[Rho]-4)\[Phi]^2-3)Cos[\[Theta]]^2+(8-4\[Rho])\[Phi]^2+4;
a3[\[Theta]_,\[Rho]_, \[Phi]_] :=(4\[Rho]-2)Cos[\[Theta]]^3+((4-4\[Rho])\[Phi]^2+3-6\[Rho])Cos[\[Theta]] + ((2\[Rho]-4)\[Phi]^2-3)Cos[\[Theta]]^2+(8+2\[Rho])\[Phi]^2+4;
a4[\[Theta]_,\[Rho]_, \[Phi]_] :=(-4 (-1+\[Rho])^2 \[Phi]^2-1+2\[Rho])Cos[\[Theta]]^3+((4\[Rho]-4)\[Phi]^2-1)Cos[\[Theta]]^2+((-2+2\[Rho])(\[Rho]-2)\[Phi]^2+1 -2\[Rho])Cos[\[Theta]] + (4-2\[Rho])\[Phi]^2+1;
a5[\[Theta]_,\[Rho]_, \[Phi]_] :=((2-2\[Rho])(\[Rho]-2)\[Phi]^2-1+2\[Rho])Cos[\[Theta]]^3+(-4 \[Phi]^2-1)Cos[\[Theta]]^2+((-2\[Rho]-4)(-1+\[Rho])\[Phi]^2+1-2\[Rho])Cos[\[Theta]]+(4\[Rho]+4)\[Phi]^2+1;
a6[\[Theta]_,\[Rho]_, \[Phi]_] :=2\[Rho](-1+\[Rho])\[Phi]^2 Cos[\[Theta]]^3-2Cos[\[Theta]]^2 \[Rho] \[Phi]^2 - 2\[Rho](-1 + \[Rho])\[Phi]^2 Cos[\[Theta]]+2\[Rho] \[Phi]^2;



ghmcLt[\[Beta]_, \[Phi]_, \[Theta]_, \[Rho]_, r_]:=(r \[Beta]^4+a0[\[Theta],\[Rho]]r^2 \[Beta]^3+(a1[\[Theta],\[Rho]]+(4-2\[Rho])\[Phi]^2)r^3 \[Beta]^2 +a2[\[Theta],\[Rho], \[Phi]]r^4 \[Beta]+a4[\[Theta],\[Rho], \[Phi]]r^5)/(\[Beta]^5+a0[\[Theta],\[Rho]]r \[Beta]^4+ (a1[\[Theta],\[Rho]]+4\[Phi]^2)r^2 \[Beta]^3 +a3[\[Theta],\[Rho], \[Phi]]r^3 \[Beta]^2+ a5[\[Theta],\[Rho], \[Phi]]r^4 \[Beta] + a6[\[Theta],\[Rho], \[Phi]]r^5);



a7[\[Theta]_]:=-Cos[\[Theta]]+2-Cos[\[Theta]]^2;
a8[\[Theta]_]:=Cos[\[Theta]]^3-Cos[\[Theta]]^2-Cos[\[Theta]]+1;


ghmcLt1Acc[\[Beta]_, \[Phi]_, \[Theta]_, r_]:=(\[Beta]^2 r+ \[Beta] r^2 a7[\[Theta]] +r^3 (Cos[\[Theta]]^3-Cos[\[Theta]]^2-Cos[\[Theta]]+2\[Phi]^2+1))/(\[Beta]^3+\[Beta]^2 r a7[\[Theta]]+\[Beta] r^2 (a8[\[Theta]]+4\[Phi]^2)+r^3 (-2 Cos[\[Theta]]^2 \[Phi]^2+2\[Phi]^2));


hmcLt[\[Beta]_,\[Phi]_, \[Rho]_, r_]:= (\[Beta]^2 r+ 2r^2 \[Beta] +  \[Phi]^2 (4-2\[Rho])r^3 +r^3)/(\[Beta]^3+2r \[Beta]^2+(4\[Phi]^2+1)r^2 \[Beta]+2 r^3 \[Phi]^2 \[Rho]);


hmcLt1Acc[\[Beta]_, \[Phi]_, r_] :=(\[Beta]^2 r+ 2\[Beta] r^2 +2r^3 \[Phi]^2+r^3)/(\[Beta]^3+2\[Beta]^2 r+\[Beta] r^2 (1+4\[Phi]^2)+r^3 2\[Phi]^2);


ighmc[\[Phi]_, \[Theta]_, \[Rho]_]:=a4[\[Theta],\[Rho], \[Phi]]/a6[\[Theta],\[Rho], \[Phi]];


ighmc1Acc[\[Phi]_, \[Theta]_]:=-((Cos[\[Theta]]^3-Cos[\[Theta]]^2-Cos[\[Theta]]+2\[Phi]^2+1)/(2\[Phi]^2 (Cos[\[Theta]]^2-1)));


ihmc[\[Phi]_, \[Rho]_]:=(2-\[Rho])/\[Rho]+1/(2\[Rho] \[Phi]^2);


ihmc1Acc[\[Phi]_] :=1+1/(2\[Phi]^2);


hmc1AccFullForm = InverseLaplaceTransform[hmcLt1Acc[\[Beta] ,\[Phi], r],\[Beta],t] //ToRadicals;
hmc1Acc[t_, \[Phi]_,r_ ] := Evaluate[hmc1AccFullForm];
hmc1AccNormalised[t_, \[Phi]_,r_ ] := hmc1Acc[t, \[Phi],r]/hmc1Acc[0, \[Phi],r];


hmcFullForm=InverseLaplaceTransform[hmcLt[\[Beta],\[Phi],\[Rho],r],\[Beta],t]//ToRadicals;
hmc[t_,\[Phi]_,\[Rho]_,r_]:=Evaluate[hmcFullForm];
hmcNormalised[t_,\[Phi]_,\[Rho]_,r_]:=hmc[t,\[Phi],\[Rho],r]/hmc[0,\[Phi],\[Rho],r];


ghmc1AccFullForm=InverseLaplaceTransform[ghmcLt1Acc[\[Beta],\[Phi],\[Theta],r],\[Beta],t]//ToRadicals;
ghmc1Acc[t_,\[Phi]_,\[Theta]_,r_]:=Evaluate[ghmc1AccFullForm];
ghmc1AccNormalised[t_,\[Phi]_,\[Theta]_,r_]=ghmc1Acc[t,\[Phi],\[Theta],r]/ghmc1Acc[0,\[Phi],\[Theta],r];


End[]; 
EndPackage[]; 