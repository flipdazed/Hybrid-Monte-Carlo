BeginPackage["mathematicaUtils`"];
cleanCString::usage="cleanCString[str_] This function cleans up with `CForm[str] // ToString` and then makes the following replacements: Cos->cos , Sin->sin, Sqrt->sqrt, Power->pow, Complex variables should be handled by a Function called Complex(x, y) that returns x + y*(complex<double> variable)";
writeFunc::usage"writeFunc[f_, name_, outDir_] This function writes a C function for f_ and assigns it the name_ and saves to outDir_. This is awesome for defining horrific Mathematica functions in C code and knowing that no typo has been made! Care should be taken to ensure the order of variables in the function is as you would expect. Chances are that they're not in the order you thought they were!";
Begin["`Private`"]; 


cleanCString[str_] := Module[{newstr},
  newstr = CForm[str] // ToString;
  newstr = StringReplace[newstr, "Cos" -> "cos"];
  newstr = StringReplace[newstr, "Sin" -> "sin"];
  newstr = StringReplace[newstr, "Sqrt" -> "sqrt"];
  newstr = StringReplace[newstr, "Power" -> "pow"];
  Return[newstr];
  ]


writeFunc[f_, name_, outDir_] := Module[
  {fnName, saveLoc, n, d, v, vars, fnVars, tmpVars, hdr, decs, e},
  
  (* generate the save name and location *)
  
  fnName = StringForm["``.hpp", name] // ToString; 
  saveLoc = FileNameJoin[{outDir, fnName}] // ToString;
  
  (*Define the Top header for the function*)
  hdr = StringJoin[{
     "#pragma once\n",
     "#ifndef MY_HEADER_H\n",
     "#define MY_HEADER_H\n\n",
     "#include <complex>\n",
     "#include \"myComplex.hpp\"\n",
     "#endif\n"}];
  
  (*Get variables from func. Make String. Delete useless bits.*)
  
  vars = DeleteDuplicates@Cases[f, _Symbol, Infinity];
  vars = Map[ToString, vars];
  If [MemberQ[vars, "E"],
   Print["E found in func"];
   ];
  vars = DeleteCases[vars, "E"];
  If [MemberQ[vars, "Null"],
   Print["Wanring! Null found in function!"];
   ];
  vars = DeleteCases[vars, "Null"];
  
  (*Order the fuinction arguments*)
  vars = Sort[vars];
  
  (*get numerator and denominators in strings of C-format*)
  
  n = Collect[Numerator[f] // N, {t, b, r, ph, p}];
  d = Collect[Denominator[f] // N, {t, b, r, ph, p}];
  n = cleanCString[n];
  d = cleanCString[d];
  
  (*define the func arguments as doubles*)
  
  fnVars = If [Length[vars] > 1,
    StringJoin[{Map[StringJoin[{" double ", # , ","}] &, Most[vars]], 
      " double ", Last[vars]}],
    StringJoin[" double ", vars]];
  Print["Found Variables:"];
  Print[fnVars];
  (*define declarations of numerator and denominators*)
  
  decs = "\n\tstd::complex<double> numerator;\n\tstd::complex<double> \
denominator;";
  (*join everything up*)
  
  hdr = StringJoin[ {hdr, "\n\ndouble ", name, "(", fnVars}, {"){\n", 
     decs,  "\n\n\tnumerator="}];
  v = StringJoin[hdr, n, {";\n\n\tdenominator="}, d, 
    {";\n\n", "\treturn (numerator/denominator).real();\n}"}
    ];
  Print[saveLoc];
  WriteString[saveLoc, v];
  ]
  
End[]; 
EndPackage[]; 