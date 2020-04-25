
t1=[6 30 78 102 126 150 167 174 198 222 246 270 294 318 342 366 390 414];
t2=[15 27 46 60 72 84 96.5 106.5 132.5 143 147 160 176 188 200 212 224 236 248 260 272 284 296 308 320 332 344 356 368 380 392 404];

ALB=[
32.4
32.2
32.7
34.4
34
34.7
28.5
31.1
28.1
30.9
31.6
31.8
35.7
35.6
34.3
34.2
34.7
33.5]';

Cr=[
52.5
51
47.6
44.9
49.9
33.9
33.3
34.3
31.4
31
36
21.9
30.6
33.5
26.2
22.8
26.3
22.7]';

OSM=[
307
295
302
291
285
286
286
286
300
280
280
277
278
298
292
277
277
271]';

ALT=[
41
36
31
39
34
30
25
28
31
34
34
37
48
46
32
35
40
68]';

TB=[
23.6
19.1
18.2
24.4
18.6
17.6
27.9
25.7
13.6
14.3
14.5
20.4
15.1
12.3
13.7
20.5
22.6
23.4]';

DB=[
7.2
6.1
5.5
6.7
5.1
3.7
8.6
6.9
3.1
4.6
4.5
3.7
5.8
3.3
5.2
7.1
8.6
9.94]';


tt = 6:0.01:414;

yy = spline(t1,ALB,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,ALB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([20,60]);
plot([6,414],[55,55],'--b')
hold on
plot([6,414],[40,40],'--k');
a=legend('sampling point of ALB','spline-inter of ALB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
ALB=yy;
 

yy = spline(t1,Cr,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,Cr,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([15,90]);
plot([6,414],[84,84],'--b')
hold on
plot([6,414],[45,45],'--k');
a=legend('sampling point of Cr','spline-inter of Cr','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
Cr=yy;


yy = spline(t1,OSM,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,OSM,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([260,310]);
plot([6,414],[301,301],'--b')
hold on
plot([6,414],[270,270],'--k');
a=legend('sampling point of OSM','spline-inter of OSM','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
OSM=yy;


yy = spline(t1,ALT,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,ALT,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([0,80]);
plot([6,414],[40,40],'--b')
hold on
plot([6,414],[7,7],'--k');
a=legend('sampling point of ALT','spline-inter of ALT','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
ALT=yy;


yy = spline(t1,TB,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,TB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([-5,30]);
plot([6,414],[23,23],'--b')
hold on
plot([6,414],[0,0],'--k');
a=legend('sampling point of TB','spline-inter of TB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
TB=yy;


yy = spline(t1,DB,tt);
yyt2=[yy(1,901) yy(1,2101) yy(1,4001) yy(1,5401) yy(1,6601) yy(1,7801) yy(1,9051) yy(1,10051) yy(1,12651) yy(1,13701) yy(1,14101) yy(1,15401) yy(1,17001) yy(1,18201) yy(1,19401) yy(1,20601) yy(1,21801) yy(1,23001) yy(1,24201) yy(1,25401) yy(1,26601) yy(1,27801) yy(1,29001) yy(1,30201) yy(1,31401) yy(1,32601) yy(1,33801) yy(1,35001) yy(1,36201) yy(1,37401) yy(1,38601) yy(1,39801)];
plot(t1,DB,'*r',tt,yy,t2,yyt2,'ok','MarkerSize',5);
hold on
xlim([0,420]);
ylim([-3,12]);
plot([6,414],[8,8],'--b')
hold on
plot([6,414],[0,0],'--k');
a=legend('sampling point of DB','spline-inter of DB','injection point of ALB','upper-bound','lower-bound');
set(a,'fontsize',25)
DB=yy;


