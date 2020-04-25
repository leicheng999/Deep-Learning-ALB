
t1=[0+6 24+6 24*2+6 24*3+6 24*4+6 24*5+6 24*6+6 24*7+6 24*8+6 24*9+6 24*10+6];
t2=[0+8 24+8 24*2+8 24*3+8 24*4+8 24*5+8 24*6+8 24*7+8 24*8+8 24*9+8 24*10+8];

ALB=[
34.3
31.6
38.2
34.9
33.8
34.3
34.3
40.3
31.5
38.9
37.3]';

Cr=[
53
55.5
65.9
51.7
43
46.5
38.7
39.4
35
33.7
36.7]';

OSM=[
275
270
264
263
253
264
259
272
274
263
268
]';

ALT=[
17
15
18
13
14
18
26
30
34
45
41
]';

TB=[
19.7
11.8
9.4
7.5
7
9.6
9.7
11.7
11.3
10.9
11.5
]';

DB=[
8
5.7
4.4
4
3
3.6
4.4
5.7
5.2
4.6
4.7
]';


tt = 6:0.01:240+8;

yy = spline(t1,ALB,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,ALB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([20,60]);
plot([6,414],[55,55],'--b')
hold on
plot([6,414],[40,40],'--k');
a=legend('sampling point of ALB','spline-inter of ALB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
ALB=yy;



yy = spline(t1,Cr,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,Cr,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([15,90]);
plot([6,414],[84,84],'--b')
hold on
plot([6,414],[45,45],'--k');
a=legend('sampling point of Cr','spline-inter of Cr','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
Cr=yy;



yy = spline(t1,OSM,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,OSM,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([240,310]);
plot([6,414],[301,301],'--b')
hold on
plot([6,414],[270,270],'--k');
a=legend('sampling point of OSM','spline-inter of OSM','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
OSM=yy;




yy = spline(t1,ALT,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,ALT,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([0,80]);
plot([6,414],[40,40],'--b')
hold on
plot([6,414],[7,7],'--k');
a=legend('sampling point of ALT','spline-inter of ALT','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
ALT=yy;



yy = spline(t1,TB,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,TB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([-5,30]);
plot([6,414],[23,23],'--b')
hold on
plot([6,414],[0,0],'--k');
a=legend('sampling point of TB','spline-inter of TB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
TB=yy;




yy = spline(t1,DB,tt);
yyt2=[yy(1,201) yy(1,201+2400*1) yy(1,201+2400*2) yy(1,201+2400*3) yy(1,201+2400*4) yy(1,201+2400*5) yy(1,201+2400*6) yy(1,201+2400*7) yy(1,201+2400*8) yy(1,201+2400*9)  yy(1,201+2400*10)]
plot(t1,DB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,250]);
ylim([-3,12]);
plot([6,414],[8,8],'--b')
hold on
plot([6,414],[0,0],'--k');
a=legend('sampling point of DB','spline-inter of DB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',15)
DB=yy;

