BeginPackage["mathematicaUtils`"];
quickVerify::usage="quickVerify[f1_, f2_, wait_: 10, maxwait_: 20, explicit_: False, 
  fullSimplify_: False] This function verifies that two expressions are equal. If it cannot do within the initial time period, it will try a faster but less accurate method. If this also fails and error message is printed.";
fullVerify::usage="fullVerify[f1_, f2_,  flipSign_: False, quotientWait_: 5, wait_: 10, maxwait_: 20] This function implements quickVerify to the numerator and denominator separately.";
getNumerator::usage="getNumerator[func_, flipSign_:False, wait_:3] gets the numerator and attempts to simplify";
getDenominator::usage="getDenominator[func_, flipSign_:False, wait_:3] gets the denominator and attempts to simplify";
maxSimplify::usage="maxSimplify[func_]:=getNumerator[func]/getDenominator[func] - see other docs";
getHiddenVars::usage="getHiddenVars[context_, varPat_] returns all variables in a given context e.g. context=packageName`Private` that match the varPat e.g. varPat = ___~~ str ~~__ ";
setPrivateVars::usage="setPrivateVars[varList_] sets all the variables returned from a list passed from getHiddenVars expects a list that can be split on the symbol ` and where the variable name is the last item in this split string and the context is all but the last item.";

Begin["`Private`"]; 

flip[bool_] :=If[bool, -1, 1]

getNumerator[func_, flipSign_:False, wait_:3]:=Collect[

flip[flipSign]*Numerator[TimeConstrained[func//FullSimplify, wait, Together[func]]],
{\[Beta],r,Cos[\[Theta]],\[Phi], \[Rho]}]//TraditionalForm

getDenominator[func_, flipSign_:False, wait_:3]:=Collect[

flip[flipSign]*Denominator[TimeConstrained[func//FullSimplify, wait, Together[func]]],
{\[Beta],r,Cos[\[Theta]],\[Phi],\[Rho]}]//TraditionalForm

maxSimplify[func_]:=getNumerator[func]/getDenominator[func];

quickVerify[f1_, f2_, wait_: 10, maxwait_: 20, explicit_: False, 
  fullSimplify_: False] := Module[{t, output, status1, status2},
  (* Try alternative method if it takes longer than wait *)
  
  t = TimeConstrained[FullSimplify[f1 == f2], wait, "Failed"];
  status1 = If[StringQ[t],
    If[explicit, (*Only prints out text if explicit flagged as True*)

          StringForm[
       "Failed to Simplify within ``s. Used PossibleZeroQ instead.", 
       wait]
      Null],
    Null];
  (* Exit calculation if it takes longer than maxwait *)
  
  t = If[StringQ[t], 
    TimeConstrained[PossibleZeroQ[n1 == n2], maxwait - wait, 
     "Failed"], t];
  status2 = If[StringQ[t],
    If[explicit,(*Only prints out text if explicit flagged as True*)
 
           StringForm[
       "Failed to determine solution within maxwait of ``s. Exited.", 
       maxwait], 
      Null]
     Null];
  output = If[SameQ[t, True], 
    {status1, status2, 
     t // VerificationTest},(*Just output test result if passed*)
    \
{status1, status2, t // VerificationTest,
     "First Function", If[fullSimplify, f1 // FullSimplify, f1],
     "Second Function", 
     If[fullSimplify, f2 // FullSimplify, f2]}(*Outputs numerator, 
    denominator if test fails*)
    ];
  Print[Column[DeleteCases[output, Null]]];
]

fullVerify[f1_, f2_,  flipSign_: False, quotientWait_: 5, wait_: 10, 
    maxwait_: 20] := Module[{n1, n2, d1, d2},
    Print["Testing Numerators..."];
    Print["... Getting first numerator"];
    n1 = getNumerator[f1, flipSign, quotientWait];
    Print["... Getting second numerator"];
    n2 = getNumerator[f2, False, quotientWait];
    Print["... Verifying"];
    quickVerify[n1, n2, wait, maxwait, True];
    Print["Testing Denominators..."];
    Print["... Getting first denominator"];
    d1 = getDenominator[f1, flipSign, quotientWait];
    Print["... Getting second denominator"];
    d2 = getDenominator[f2, False, quotientWait];
    Print["... Verifying"];
    quickVerify[d1, d2, wait, maxwait, True];
    ]

getHiddenVars[context_, varPat_] := 
  Select[Names[StringJoin[context, "*"]], 
   StringMatchQ[#, varPat] &];

setPrivateVars[varList_] :=
 Module[{packageName, varName, contexts, vars, var, expr, i},
  contexts =  StringSplit[varList[[1]], "`"];
  packageName = 
   StringJoin[Map[StringJoin[#, "`"] &, Most[contexts]]];
  vars = varList /. x_ :> StringReplace[x, packageName -> "" ];
  For[i = 1, i < Length[vars] + 1, i++,
   var = vars[[i]];
   expr = StringJoin[var, "=", packageName, var];
   Clear[ToExpression[var]];
   ToExpression [expr];
   Print["Set: ", var, " = ", expr ];
   ]
]
End[]; 
EndPackage[]; 