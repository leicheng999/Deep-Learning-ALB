


output_x=[14.8, 15, 15.1, 15.4, 15.5, 15.7]

output_y=[0.0, 0.0, 20.0, 20.0, 0.0, 0.0]

tt = 14.8:0.01:15.7;

yy = interp1(output_x,output_y,tt);
plot(tt,yy);
hold on
xlim([14.8,15.7]);
ylim([-3,22]);
a=legend('y(t):g/h')
set(a,'fontsize',17)
intake=yy;

