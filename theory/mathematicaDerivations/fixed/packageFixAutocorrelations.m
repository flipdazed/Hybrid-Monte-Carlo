(* ::Package:: *)

BeginPackage["packageFixAutocorrelations`"];
ghmcLt::usage = "ghmcLt[\[Beta]_, \[Phi]_, \[Theta]_, \[Rho]_, r_] is the Laplace transformed GHMC autocorrelation function"; 
ghmcLt1Acc::usage = "ghmcLt1Acc[\[Beta]_, \[Phi]_, \[Theta]_, r_] is the Laplace transformed GHMC autocorrelation function with unit acceptance probability"; 
ghmcLtCoeffs::usage="ghmcLtCoeffs[z_, \[Phi]_, \[Theta]_, \[Rho]_, \[Tau]_] is the same as ghmcLt[\[Beta]_, \[Phi]_, \[Theta]_, \[Rho]_, \[Tau]_] 
	if z=Exp[-\[Beta] \[Tau]] it allows the coefficients to be more easily extracted";
hmcLt::usage = "hmcLt[\[Beta]_, \[Phi]_, \[Rho]_, r_] is the Laplace transformed HMC autocorrelation function"; 
hmcLt1Acc::usage = "hmcLt1Acc[\[Beta]_, \[Phi]_, r_] is the Laplace transformed HMC autocorrelation function with unit acceptance probability"; 
ighmc::usage = "ighmc[\[Phi]_, \[Theta]_, \[Rho]_] is the GHMC integrated autocorrelation function (note this is the inverted Laplace Transform)"; 
ighmc1Acc::usage = "ighmc[\[Phi]_, \[Theta]_] is the GHMC integrated autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
ihmc::usage = "ihmc[\[Phi]_, \[Rho]_] is the HMC integrated autocorrelation function (note this is the inverted Laplace Transform)"; 
ihmc1Acc::usage = "ihmc1Acc[\[Phi]_] is the HMC integrated autocorrelation function with unit acceptance probability (note this is the inverted Laplace Transform)"; 
Begin["`Private`"]; 


a0[\[Rho]_,\[Phi]_]:=-\[Rho] Cos[\[Phi]]^2-1+\[Rho];
a1[\[Theta]_,\[Rho]_] := (-1+2\[Rho])Cos[\[Theta]]^3;
a2[\[Theta]_, \[Rho]_,\[Phi]_] := (-2\[Rho]+1+2\[Rho] Cos[\[Phi]]^2)Cos[\[Theta]]^2;
a3[\[Theta]_, \[Rho]_,\[Phi]_]:=(\[Rho]+\[Rho] Cos[\[Phi]]^2-1)Cos[\[Theta]];
a4[\[Theta]_, \[Rho]_,\[Phi]_]:=(-2\[Rho] Cos[\[Phi]]^2+1)Cos[\[Theta]]+(Cos[\[Theta]]^2+1)a0[\[Rho],\[Phi]];
a5[\[Theta]_, \[Rho]_,\[Phi]_]:=a3[\[Theta], \[Rho],\[Phi]]Cos[\[Theta]]^2;
a6[\[Theta]_, \[Rho]_]:=(1-2\[Rho])Cos[\[Theta]]^3;


ghmcLt[\[Beta]_, \[Phi]_, \[Theta]_, \[Rho]_, \[Tau]_]:=(-(Exp[- \[Beta] \[Tau] ] a0[\[Rho],\[Phi]]-Exp[-3\[Beta] \[Tau]]a1[\[Theta], \[Rho]]+Exp[-2\[Beta] \[Tau]]a2[\[Theta], \[Rho],\[Phi]]+Exp[-2\[Beta] \[Tau]]a3[\[Theta], \[Rho],\[Phi]]))/(a4[\[Theta], \[Rho],\[Phi]]Exp[-\[Beta] \[Tau]]+a5[\[Theta], \[Rho],\[Phi]]Exp[-2\[Beta] \[Tau]]+a6[\[Theta],\[Rho]]Exp[-3\[Beta] \[Tau]]+Exp[-2\[Beta] \[Tau]](a2[\[Theta], \[Rho],\[Phi]]+a3[\[Theta], \[Rho],\[Phi]])+1);


ghmcLtCoeffs[z_, \[Phi]_, \[Theta]_, \[Rho]_, \[Tau]_]:= -(z a0[\[Rho],\[Phi]]-z^3 a1[\[Theta], \[Rho]]+z^2 a2[\[Theta], \[Rho],\[Phi]]+z^2 a3[\[Theta], \[Rho],\[Phi]])/(a4[\[Theta], \[Rho],\[Phi]]z+a5[\[Theta], \[Rho],\[Phi]]z^2+a6[\[Theta],\[Rho]]z^3+z^2 (a2[\[Theta], \[Rho],\[Phi]]+a3[\[Theta], \[Rho],\[Phi]])+1);



a7[\[Theta]_, \[Phi]_]:=(-1+2Cos[\[Phi]]^2)Cos[\[Theta]]^2;
a8[\[Theta]_, \[Phi]_]:=(Cos[\[Phi]]^2 Cos[\[Theta]]^2+(-1+2Cos[\[Phi]]^2)Cos[\[Theta]]+Cos[\[Phi]]^2);
a9[\[Theta]_, \[Phi]_]:=a7[\[Theta], \[Phi]]+Cos[\[Phi]]^2 Cos[\[Theta]];


ghmcLt1Acc[\[Beta]_, \[Phi]_, \[Theta]_, \[Tau]_]:=(-(Exp[-\[Beta] \[Tau]]Cos[\[Phi]]^2+Exp[-3\[Beta] \[Tau]]Cos[\[Theta]]^3-Exp[-2\[Beta] \[Tau]]a7[\[Theta], \[Phi]]-Exp[-2\[Beta] \[Tau]]Cos[\[Phi]]^2 Cos[\[Theta]]))/(Exp[-\[Beta] \[Tau]]a8[\[Theta], \[Phi]]+(Exp[-3\[Beta] \[Tau]]-Exp[-2\[Beta] \[Tau]]Cos[\[Phi]]^2)Cos[\[Theta]]^3-Exp[-2\[Beta] \[Tau]]a9[\[Theta], \[Phi]]-1);


hmcLt[\[Beta]_,\[Phi]_, \[Rho]_, \[Tau]_]:= -a0[\[Rho],\[Phi]]/(Exp[\[Beta] \[Tau]]+a0[\[Rho],\[Phi]]);


hmcLt1Acc[\[Beta]_, \[Phi]_, \[Tau]_] :=Cos[\[Phi]]^2/(Exp[\[Beta] \[Tau]]-Cos[\[Phi]]^2);


ighmc[\[Phi]_, \[Theta]_, \[Rho]_]:=-(((1-2\[Rho] )Cos[\[Theta]]^2+2Cos[\[Theta]]\[Rho] Cos[\[Phi]]^2-\[Rho] Cos[\[Phi]]^2-1+\[Rho])/((Cos[\[Theta]]-1)(Cos[\[Theta]]+1)(Cos[\[Phi]]-1)(Cos[\[Phi]]+1)\[Rho]));


ighmc1Acc[\[Phi]_, \[Theta]_]:=(Cos[\[Theta]]^2-2Cos[\[Theta]]Cos[\[Phi]]^2+Cos[\[Phi]]^2)/(Cos[\[Theta]]^2 Cos[\[Phi]]^2-Cos[\[Theta]]^2-Cos[\[Phi]]^2+1);


ihmc[\[Phi]_, \[Rho]_]:=-a0[\[Rho],\[Phi]]/(\[Rho](1- Cos[\[Phi]])(1+Cos[\[Phi]]));


ihmc1Acc[\[Phi]_] :=Cos[\[Phi]]^2/(1-Cos[\[Phi]]^2);


End[]; 
EndPackage[]; 
