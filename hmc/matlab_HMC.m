% EXAMPLE 1: SIMULATING HAMILTONIAN DYNAMICS
%            OF HARMONIC OSCILLATOR
% STEP SIZE
delta = 0.1;
 
% # LEAP FROG
L = 70;
 
% DEFINE KINETIC ENERGY FUNCTION
K = inline('p^2/2','p');
 
% DEFINE POTENTIAL ENERGY FUNCTION FOR SPRING (K =1)
U = inline('1/2*x^2','x');
 
% DEFINE GRADIENT OF POTENTIAL ENERGY
dU = inline('x','x');
 
% INITIAL CONDITIONS
x0 = -4; % POSTIION
p0 = 1;  % MOMENTUM
figure
 
%% SIMULATE HAMILTONIAN DYNAMICS WITH LEAPFROG METHOD
% FIRST HALF STEP FOR MOMENTUM
pStep = p0 - delta/2*dU(x0)';
 
% FIRST FULL STEP FOR POSITION/SAMPLE
xStep = x0 + delta*pStep;
 
% FULL STEPS
for jL = 1:L-1
    % UPDATE MOMENTUM
    pStep = pStep - delta*dU(xStep);
 
    % UPDATE POSITION
    xStep = xStep + delta*pStep;
 
    % UPDATE DISPLAYS
    subplot(211), cla
    hold on;
    xx = linspace(-6,xStep,1000);
    plot(xx,sin(6*linspace(0,2*pi,1000)),'k-');
    plot(xStep+.5,0,'bo','Linewidth',20)
    xlim([-6 6]);ylim([-1 1])
    hold off;
    title('Harmonic Oscillator')
    subplot(223), cla
    b = bar([U(xStep),K(pStep);0,U(xStep)+K(pStep)],'stacked');
    set(gca,'xTickLabel',{'U+K','H'})
    ylim([0 10]);
    title('Energy')
    subplot(224);
    plot(xStep,pStep,'ko','Linewidth',20);
        xlim([-6 6]); ylim([-6 6]); axis square
    xlabel('x'); ylabel('p');
    title('Phase Space')
    pause(.1)
end
% (LAST HALF STEP FOR MOMENTUM)
pStep = pStep - delta/2*dU(xStep);