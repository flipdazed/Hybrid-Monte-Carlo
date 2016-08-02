Mathematica Derivations
===============

## Table of Contents
 .  [Calling Functions from Python](#acorr)
 .  [Integrated Autocorrelation Functions](#iacorr)
 .  [Molecular Dynamics](#mdmc)
 
<a name="mdmc"/>
## Calling Functions from Python

These instructions are adapted from [this Stack Exchange link](http://mathematica.stackexchange.com/a/4673/41800)

1. Open `Wolfram Mathematica` and evaluate a cell containing:
    
        Part[$CommandLine, 1]
    
    For me this location is:
    
        /Applications/Mathematica.app/Contents/MacOS/WolframKernel
    
1. Create a script named `runMath` (no extension) with the content:
    
        #!/Applications/Mathematica.app/Contents/MacOS/WolframKernel -script
        
        value=$CommandLine;
        For[i = 1, i < Length[value]+1, i++,
            str = StringForm["item: `` Value: ``\n", i, Part[value,i]];
            WriteString[$Output, str];
        ];
    
    Note that the SheBang will vary depending on the output of step 1.

3. Give execution privileges to the file.
    
        sudo chmod +x runMath

4. Enter the following into the terminal:
    
        ./runMath "Select This number"
    
    Which should return something like:
    
        item: 1 Value: /Applications/Mathematica.app/Contents/MacOS/WolframKernel
        item: 2 Value: -script
        item: 3 Value: ./runMath
        item: 4 Value: Select This number
    
5. Open the script named `runMath` again and replace the contents with:

        #!/Applications/Mathematica.app/Contents/MacOS/WolframKernel -script
        
        value=ToExpression[$CommandLine[[4]]];
        Print[value];
    
    where `4` is the number the script instructed me to replace.

6. Now this can now be called from python using `pyMathematica`:

        from pyMathematica import eval
        eval('Exp[4] // N')
<a name="acorr"/>
## Autocorrelation Functions
Note: integrated autocorrelations are also called from here by the `.integrated()` method

 .  Magnetic Susceptibility, $\phi^2$
  * Fixed length trajectories
  * Exponentially distributed trajectories
