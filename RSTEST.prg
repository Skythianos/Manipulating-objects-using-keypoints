1 ' RemoteSurf program
2 ' Ayhan Daniel
3 Ovrd 20
4 Servo On
5 pcurr = P_Curr
6 p1 = pcurr
7 *c1
8 M_Out16(10160) = pcurr.X ' PC: 1000
9 M_Out16(10176) = pcurr.Y ' PC: 1001
10 M_Out16(10192) = pcurr.Z ' PC: 1002
11 M_Out16(10208) = pcurr.A * 57.29 ' PC: 1003
12 M_Out16(10224) = pcurr.B * 57.29 ' PC: 1004
13 M_Out16(10240) = pcurr.C * 57.29 ' PC: 1005
14 M_Out16(10816) = ecounter ' PC: 1041
15 ex = M_In16(10160) ' PC: 500
16 ey = M_In16(10176) ' PC: 501
17 ez = M_In16(10192) ' PC: 502
18 ea = M_In16(10208) ' PC: 503
19 eb = M_In16(10224) ' PC: 504
20 ec = M_In16(10240) ' PC: 505
21 ecounter = M_In16(10800) ' PC: 540
22 p1.X = ex
23 p1.Y = ey
24 p1.Z = ez
25 p1.A = ea / 57.296
26 p1.B = eb / 57.296
27 p1.C = ec / 57.296
28 Mov p1
29 Dly 0.5
30 GoTo *c1
pcurr=(+412.90,+1.26,+208.93,+35.34,+90.00,+35.34)(6,0)
p1=(+412.90,+1.26,+208.93,+0.00,+0.00,+0.00,+0.00,+0.00)(6,0)
p_szerel=(0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00)(,)
