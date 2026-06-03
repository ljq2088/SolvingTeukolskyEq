clear
% close all
format long
% parameters
M = 1; la = 4*M; 
l = 0;
omega = 1e-3/M;
rp = 3*M + omega^(-1/2) ; 
zp = 2*M/rp; xp = 2*M*(1/zp+log(1-zp)-log(zp));

N = 128;

AnMR_flag = omega < 1e-1;

if AnMR_flag ==0
    % [1,zp] 倒序,z1从1到zp
    [Dz1,z1] = cheb(N); z1 = (1-zp)/2 * (z1+1) + zp; Dz1 = Dz1 / ((1-zp)/2); D2z1 = Dz1^2;
    % [zp,0]
    [Dz0,z0] = cheb(N); z0 = (zp-0)/2 * (z0+1); Dz0 = Dz0 / ((zp-0)/2); D2z0 = Dz0^2;
else
    [Dy,y] = cheb(N);
    kappa = abs(log(omega*2*M));
    kappa_in = kappa/2;

    % y\in[-1,1] -> z1\in[1,zp]
    y2z1 = @(y) (1-zp) * sinh( kappa_in*(y + 1)/2 ) / sinh(kappa_in) + zp;
    dz1dy= (1-zp) * kappa_in/2 * cosh(kappa_in*(y + 1)/2) / sinh(kappa_in);
    z1 = y2z1(y); Dz1 = Dy ./ dz1dy; D2z1 = Dz1^2;

    % y\in[-1,1] -> z0\in[zp,0]
    y2z0 = @(y) zp * sinh( kappa*(y + 1)/2 ) / sinh(kappa);
    dz0dy= zp * kappa/2 * cosh(kappa*(y + 1)/2) / sinh(kappa);
    z0 = y2z0(y); Dz0 = Dy ./ dz0dy; D2z0 = Dz0^2;
end

s = -1i*omega*4*M;
rh = 2*M;
% solving 'in' & 'down' solution in ingoing coordinates
B1 = Bondi(z1,Dz1,D2z1,l,s);
% normalized condition: phi_in(1) = 1
tmp = B1(end,:)==B1(end,1);
B1(end,:) = tmp;
phi_in = B1\flip(tmp.');
dphi_in = Dz1*phi_in;

B0 = Bondi(z0,Dz0,D2z0,l,s);
% normalized condition: phi_down(end) = 1
tmp = B0(1,:)==B0(1,end);
B0(1,:) = tmp; 
phi_down = B0\flip(tmp.');
dphi_down = Dz0*phi_down;

% res_down = calculate_residual(z0,Dz0,D2z0,l,s,phi_down);
% plot(res_down)
% set(gca,'yscale','log')

% solve linear GF
tmp = exp(-1i*omega*xp);
dxp = rh/(zp^2*(zp-1));
GFM11 = tmp*phi_down(1);
GFM21 = tmp*(dphi_down(1)+phi_down(1)*(-1i*omega)*dxp);
GFM = [GFM11,conj(GFM11);...
       GFM21,conj(GFM21)];
C = GFM\[ phi_in(end)*tmp ; tmp*(dphi_in(end)+phi_in(end)*(-1i*omega)*dxp) ];

Cid = C(1); Ciu = C(2);

T = 1/abs(Cid)^2; %
R = abs(Ciu)^2/abs(Cid)^2; %

figure;
spec_plot(abs(phi_down))
hold on
spec_plot(abs(phi_in))
% Wronskian
W = 1i*2*omega*Cid;
x_fun = @(z) rh.*(1./z + log(1-z) - log(z));

%% cheb_interpolate 
% [1,zp]
phi_in_ReCheb = real_to_cheb(real(phi_in)); 
phi_in_ImCheb = real_to_cheb(imag(phi_in));
phi_down_ReCheb = real_to_cheb(real(phi_down));
phi_down_ImCheb = real_to_cheb(imag(phi_down));
if AnMR_flag ==0
    phi_in_fun1 = @(z) ( cheb_interpolate(phi_in_ReCheb,zp,1,z) + ...
                      1i*cheb_interpolate(phi_in_ImCheb,zp,1,z) ).*exp(-1i*omega*x_fun(z));
    phi_out_fun1 = @(z) conj(phi_in_fun1(z));
    phi_up_fun1 = @(z) -conj(Ciu)*phi_in_fun1(z) + Cid*phi_out_fun1(z);
    
    % [zp,0]
    
    phi_down_fun0 = @(z) (cheb_interpolate(phi_down_ReCheb,0,zp,z) + ...
                      1i*cheb_interpolate(phi_down_ImCheb,0,zp,z)).*exp(-1i*omega*x_fun(z));
    phi_up_fun0 = @(z) conj(phi_down_fun0(z));
    phi_in_fun0 = @(z) Cid*phi_down_fun0(z) + Ciu*phi_up_fun0(z);
else % 谱系数是y坐标对应的谱系数
    z12y = @(z) asinh((z-zp)/(1-zp)*sinh(kappa_in))*2/kappa_in - 1;
    % 先在y坐标上构造插值函数
    phi_in_funy = @(y) cheb_interpolate(phi_in_ReCheb,-1,1,y) + 1i*cheb_interpolate(phi_in_ImCheb,-1,1,y);
    % 利用中z2y变换
    phi_in_fun1 = @(z) phi_in_funy(z12y(z)).*exp(-1i*omega*x_fun(z));
    phi_out_fun1 = @(z) conj(phi_in_fun1(z));
    phi_up_fun1 = @(z) -conj(Ciu)*phi_in_fun1(z) + Cid*phi_out_fun1(z);
    
    z02y = @(z) asinh(z/zp*sinh(kappa))*2/kappa - 1;
    phi_down_funy = @(y) cheb_interpolate(phi_down_ReCheb,-1,1,y) + 1i*cheb_interpolate(phi_down_ImCheb,-1,1,y);
    phi_down_fun0 = @(z) phi_down_funy(z02y(z)).*exp(-1i*omega*x_fun(z));
    phi_up_fun0 = @(z) conj(phi_down_fun0(z));
    phi_in_fun0 = @(z) Cid*phi_down_fun0(z) + Ciu*phi_up_fun0(z);
end
phi_in_fun = @(z)phi_in_fun_full(z,zp,phi_in_fun0,phi_in_fun1);
phi_up_fun = @(z)phi_up_fun_full(z,zp,phi_up_fun0,phi_up_fun1);

% %loglog(rh./z0,abs(phi_in_fun0(z0)))
tmp = [linspace(rh+0.01, 10*rp, 800), 10.^linspace(1+log10(rp), 6-log10(omega), 500)];
%p = loglog(tmp*omega,abs(phi_in_fun(rh./tmp)),LineWidth=1,DisplayName=num2str(omega)); %.*(rh./tmp).^2
% xlabel("$\omega r$","Interpreter","latex",FontSize=20)
% ylabel("$\phi_{\rm{in}}$","Interpreter","latex",FontSize=20)

loglog(tmp*omega ,abs(phi_up_fun(rh./tmp)),LineWidth=1,DisplayName=num2str(omega))
xlabel("$\omega r$","Interpreter","latex",FontSize=20)
ylabel("$\phi_{\rm{up}}$","Interpreter","latex",FontSize=20)
% plot(z1 ,abs(phi_up_fun1(z1)))
hold on
%yline(abs(Cid),'--')
%loglog(rh./z1,abs(phi_in_fun1(z1)))
%legend
abs(Cid)
%% integrate
integrand1 = @(z) abs(phi_in_fun1(z)).^2.*phi_in_fun1(z).^2;
[val1,errbnd1] = quadgk(integrand1,zp,1,"RelTol",1e-10);
val1 = val1/rh;
% consistent with variable substitution
% fun = @(x) 1- exp(-x);
% tmp_fun = @(x) integrand1(fun(x)).*exp(-x);
% [valtmp,errtmp]=quadgk(tmp_fun,-log(1-zp),37,"RelTol",1e-10);

% transform to r coordinate
integrand0 = @(r) abs(phi_in_fun0(rh./r)).^2.*phi_in_fun0(rh./r).^2 ./ r.^2;
[val0,errbnd0] = quadgk(integrand0,rh/zp,Inf,'MaxIntervalCount',2e5);

% tmpi = @(z) exp(-1i*2*omega*x_fun(z)) .* phi_down_fun0(z).^2 .*abs(phi_down_fun0(z)).^2;
% [val0,errbnd0]=quadgk(tmpi,0,zp,"RelTol",1e-10);

A1out = -(val1+val0)*Cl/W;
% rel err
errbnd1 = abs(errbnd1/val1);
errbnd0 = abs(errbnd0/val0);


integrand1 = @(z) abs(phi_in_fun1(z)).^2.*phi_in_fun1(z).*phi_up_fun1(z);
[val1,errbnd11] = quadgk(integrand1,zp,1);
val1 = val1/rh;
% test omega dependence
% integrand0 = @(r) abs(phi_in_fun0(rh./r)).^2 .*phi_in_fun0(rh./r) ./ r.^2;
integrand0 = @(r) abs(phi_in_fun0(rh./r)).^2.*phi_in_fun0(rh./r).*phi_up_fun0(rh./r)./ r.^2;
[val0,errbnd00] = quadgk(integrand0,rh/zp,Inf,'MaxIntervalCount',2e5);
A1in = -(val1+val0)*Cl/W;

% rel err
errbnd11 = abs(errbnd11/val1);
errbnd00 = abs(errbnd00/val0);

% nonlinear GF 
T1 = 2*real(A1in)/abs(Cid)^2;
R1 = 2*real(Ciu*conj(A1out))/abs(Cid)^2;

function val = phi_in_fun_full(z,zp,phi_in_fun0,phi_in_fun1)
    val = 0*z;
    val(z<zp) = phi_in_fun0(z(z<zp));
    val(z>=zp) = phi_in_fun1(z(z>=zp));     
end
function val = phi_up_fun_full(z,zp,phi_up_fun0,phi_up_fun1)
    val = 0*z;
    val(z<zp) = phi_up_fun0(z(z<zp));
    val(z>=zp) = phi_up_fun1(z(z>=zp));     
end

function res = calculate_residual(z,D1,D2,l,s,phi)
    a2 = z.^2 .* (1-z);
    a1 = z.*(2-3*z) - s;
    a0 = -(l*(l+1) + z);
    B = a2.*D2 + a1.*D1 + diag(a0);

    res = abs(B*phi)./max([abs(a2.*D2*phi),abs(a1.*D1*phi),abs(a0.*phi)],[],2);
    res = min([abs(B*phi),res],[],2);
end